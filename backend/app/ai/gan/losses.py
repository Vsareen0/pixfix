"""Training losses for the PixFix GAN.

Total generator loss = λ_hole·L1(hole) + λ_valid·L1(valid) + λ_adv·hinge_G
                       [+ λ_perc·perceptual (VGG16, optional)]
Discriminator loss   = hinge_D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l1(pred, target, mask):
    """Mean absolute error restricted to pixels where mask == 1."""
    denom = mask.sum() * pred.shape[1] + 1e-6
    return (torch.abs(pred - target) * mask).sum() / denom


def reconstruction_loss(raw, target, mask, w_hole=6.0, w_valid=1.0):
    return w_hole * masked_l1(raw, target, mask) + w_valid * masked_l1(raw, target, 1 - mask)


def d_hinge_loss(real_scores, fake_scores):
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def g_hinge_loss(fake_scores):
    return -fake_scores.mean()


class PerceptualLoss(nn.Module):
    """L1 distance between VGG16 feature maps (relu1_2, relu2_2, relu3_3).

    Needs torchvision's ImageNet VGG16 weights (downloaded on first use).
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList([feats[:4], feats[4:9], feats[9:16]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # inputs in [-1, 1] -> ImageNet normalisation
        x = ((pred + 1) / 2 - self.mean) / self.std
        y = ((target + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss
