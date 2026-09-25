"""Train the PixFix inpainting GAN.

Usage (from backend/):
    python -m app.ai.gan.train --data /path/to/images --epochs 20 --batch-size 8
    python -m app.ai.gan.train --data ./data/train --val-data ./data/val --perceptual
    python -m app.ai.gan.train --data ./data/train --resume checkpoints/last.pth

Outputs (in --out, default ./checkpoints):
    last.pth           full checkpoint (G, D, optimisers) for resuming
    best.pth           same, lowest validation L1
    pixfix_gan.pth     generator-only weights for the web app (copy to backend/models/)
    samples/epoch_XXX.png   [masked input | output | ground truth] grids
    history.csv        per-epoch losses and validation PSNR
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import InpaintingDataset
from .losses import PerceptualLoss, d_hinge_loss, g_hinge_loss, reconstruction_loss
from .networks import InpaintGenerator, PatchDiscriminator, count_parameters


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the PixFix inpainting GAN")
    p.add_argument("--data", required=True, help="folder of training images (searched recursively)")
    p.add_argument("--val-data", help="optional validation folder (default: 5%% split of --data)")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--base", type=int, default=32, help="generator width")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=4e-4)
    p.add_argument("--w-adv", type=float, default=0.1)
    p.add_argument("--w-perc", type=float, default=0.1)
    p.add_argument("--perceptual", action="store_true", help="add VGG16 perceptual loss")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke tests)")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args(argv)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr(pred, target):
    """PSNR in dB for tensors in [-1, 1]."""
    mse = torch.mean(((pred + 1) / 2 - (target + 1) / 2) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def save_samples(path, img, mask, comp, n=4):
    from torchvision.utils import save_image

    n = min(n, img.shape[0])
    masked = img[:n] * (1 - mask[:n]) + mask[:n]  # holes shown white
    grid = torch.cat([masked, comp[:n], img[:n]], dim=0)
    save_image((grid + 1) / 2, path, nrow=n)


@torch.no_grad()
def validate(G, loader, device, max_batches=20):
    G.eval()
    l1_sum, psnr_sum, n = 0.0, 0.0, 0
    for i, (img, mask) in enumerate(loader):
        if i >= max_batches:
            break
        img, mask = img.to(device), mask.to(device)
        _, comp = G(img, mask)
        l1_sum += torch.mean(torch.abs(comp - img)).item()
        psnr_sum += psnr(comp, img)
        n += 1
    G.train()
    return (l1_sum / max(n, 1), psnr_sum / max(n, 1))


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    full = InpaintingDataset(args.data, size=args.size, train=True)
    if args.val_data:
        train_ds = full
        val_ds = InpaintingDataset(args.val_data, size=args.size, train=False, fixed_masks=True)
    else:
        n_val = max(1, int(len(full) * 0.05)) if len(full) > 1 else 0
        if n_val:
            train_ds, val_ds = random_split(full, [len(full) - n_val, n_val],
                                            generator=torch.Generator().manual_seed(0))
        else:
            train_ds, val_ds = full, full

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                          num_workers=args.workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    G = InpaintGenerator(base=args.base).to(device)
    D = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    perc = PerceptualLoss().to(device) if args.perceptual else None

    start_epoch, best_val, step = 0, float("inf"), 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        G.load_state_dict(ck["generator"]); D.load_state_dict(ck["discriminator"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start_epoch, best_val, step = ck["epoch"] + 1, ck.get("best_val", best_val), ck.get("step", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Device: {device} | train images: {len(train_ds)} | val images: {len(val_ds)}")
    print(f"Generator params: {count_parameters(G):,} | Discriminator params: {count_parameters(D):,}")

    history_path = out / "history.csv"
    new_history = not history_path.exists() or not args.resume
    hist_f = open(history_path, "w" if new_history else "a", newline="")
    hist = csv.writer(hist_f)
    if new_history:
        hist.writerow(["epoch", "step", "loss_d", "loss_g_adv", "loss_rec", "loss_perc",
                       "val_l1", "val_psnr", "seconds"])

    cfg = {"base": args.base, "input_size": args.size}
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        sums = {"d": 0.0, "adv": 0.0, "rec": 0.0, "perc": 0.0}
        n = 0
        for img, mask in train_dl:
            img, mask = img.to(device), mask.to(device)

            # ---- Discriminator step: real vs. generated (composited) images
            with torch.no_grad():
                _, comp = G(img, mask)
            loss_d = d_hinge_loss(D(img, mask), D(comp, mask))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- Generator step: reconstruction + fool D (+ perceptual)
            raw, comp = G(img, mask)
            loss_rec = reconstruction_loss(raw, img, mask)
            loss_adv = g_hinge_loss(D(comp, mask))
            loss_g = loss_rec + args.w_adv * loss_adv
            loss_p = torch.zeros((), device=device)
            if perc is not None:
                loss_p = perc(comp, img)
                loss_g = loss_g + args.w_perc * loss_p
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            sums["d"] += loss_d.item(); sums["adv"] += loss_adv.item()
            sums["rec"] += loss_rec.item(); sums["perc"] += loss_p.item()
            n += 1; step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}: D={loss_d.item():.3f} "
                      f"G_adv={loss_adv.item():.3f} rec={loss_rec.item():.3f}")
            if args.max_steps and step >= args.max_steps:
                break

        val_l1, val_psnr = validate(G, val_dl, device)
        secs = time.time() - t0
        avg = {k: v / max(n, 1) for k, v in sums.items()}
        print(f"== epoch {epoch}: D={avg['d']:.3f} G_adv={avg['adv']:.3f} rec={avg['rec']:.3f} "
              f"val_L1={val_l1:.4f} val_PSNR={val_psnr:.2f}dB ({secs:.0f}s)")
        hist.writerow([epoch, step, f"{avg['d']:.5f}", f"{avg['adv']:.5f}", f"{avg['rec']:.5f}",
                       f"{avg['perc']:.5f}", f"{val_l1:.5f}", f"{val_psnr:.3f}", f"{secs:.1f}"])
        hist_f.flush()

        # samples
        img, mask = next(iter(val_dl))
        G.eval()
        with torch.no_grad():
            _, comp = G(img.to(device), mask.to(device))
        G.train()
        save_samples(out / "samples" / f"epoch_{epoch:03d}.png", img, mask, comp.cpu())

        ck = {"generator": G.state_dict(), "discriminator": D.state_dict(),
              "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
              "epoch": epoch, "step": step, "best_val": min(best_val, val_l1), "config": cfg}
        torch.save(ck, out / "last.pth")
        torch.save({"generator": G.state_dict(), "config": cfg}, out / "pixfix_gan.pth")
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(ck, out / "best.pth")
            torch.save({"generator": G.state_dict(), "config": cfg}, out / "pixfix_gan_best.pth")

        if args.max_steps and step >= args.max_steps:
            break

    hist_f.close()
    print(f"Done. Copy {out / 'pixfix_gan_best.pth'} to backend/models/pixfix_gan.pth to use it in the app.")


if __name__ == "__main__":
    main()
