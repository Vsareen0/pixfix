"""Download the pretrained LaMa weights (TorchScript, ~196 MB) into backend/models/.

    python scripts/download_models.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
DEST = Path(__file__).resolve().parent.parent / "models" / "big-lama.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"Already present: {DEST}")
        return
    tmp = DEST.with_suffix(".part")
    print(f"Downloading {URL}\n -> {DEST}")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r{pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, tmp, progress)
    print()
    digest = sha256(tmp)
    if digest != SHA256:
        tmp.unlink(missing_ok=True)
        sys.exit(f"Checksum mismatch ({digest}); download aborted.")
    tmp.rename(DEST)
    print("Done.")


if __name__ == "__main__":
    main()
