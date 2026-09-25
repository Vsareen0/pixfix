"""Module 4: The AI Core – PixFix Inpainting GAN.

Generator: encoder–decoder built from *gated convolutions* (Yu et al., "Free-Form
Image Inpainting with Gated Convolution", ICCV 2019) with a dilated-convolution
bottleneck for a large receptive field (Iizuka et al., "Globally and Locally
Consistent Image Completion", SIGGRAPH 2017).

Discriminator: SN-PatchGAN – a fully convolutional discriminator with spectral
normalisation that scores many overlapping patches as real/fake. Used only
during training.

Input convention: image in [-1, 1], mask in {0, 1} with 1 = hole.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GatedConv2d(nn.Module):
    """conv_feature * sigmoid(conv_gate). The gate learns a soft, per-pixel,
    per-channel validity mask, which handles free-form holes better than
    vanilla or partial convolutions."""

    def __init__(self, cin, cout, k=3, stride=1, dilation=1, activation=True):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.feature = nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation)
        self.gate = nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation)
        self.act = nn.ELU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.feature(x)) * torch.sigmoid(self.gate(x))


class GatedUpConv2d(nn.Module):
    """Nearest-neighbour upsample followed by gated conv (avoids checkerboard artefacts)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = GatedConv2d(cin, cout, 3)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class InpaintGenerator(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c = base
        self.encoder = nn.Sequential(
            GatedConv2d(4, c, 5),                 # H
            GatedConv2d(c, 2 * c, 3, stride=2),   # H/2
            GatedConv2d(2 * c, 2 * c, 3),
            GatedConv2d(2 * c, 4 * c, 3, stride=2),  # H/4
            GatedConv2d(4 * c, 4 * c, 3),
            GatedConv2d(4 * c, 4 * c, 3),
        )
        self.bottleneck = nn.Sequential(
            GatedConv2d(4 * c, 4 * c, 3, dilation=2),
            GatedConv2d(4 * c, 4 * c, 3, dilation=4),
            GatedConv2d(4 * c, 4 * c, 3, dilation=8),
            GatedConv2d(4 * c, 4 * c, 3, dilation=16),
            GatedConv2d(4 * c, 4 * c, 3),
            GatedConv2d(4 * c, 4 * c, 3),
        )
        self.decoder = nn.Sequential(
            GatedUpConv2d(4 * c, 2 * c),          # H/2
            GatedConv2d(2 * c, 2 * c, 3),
            GatedUpConv2d(2 * c, c),              # H
            GatedConv2d(c, c // 2, 3),
            nn.Conv2d(c // 2, 3, 3, padding=1),
        )

    def forward(self, image, mask):
        """image: Bx3xHxW in [-1,1]; mask: Bx1xHxW (1 = hole). H, W divisible by 4.

        Returns (raw_output, composited_output) where the composite keeps the
        known pixels from the input and only fills the hole.
        """
        masked = image * (1 - mask)
        x = torch.cat([masked, mask], dim=1)
        x = self.encoder(x)
        x = self.bottleneck(x)
        raw = torch.tanh(self.decoder(x))
        comp = raw * mask + image * (1 - mask)
        return raw, comp


class PatchDiscriminator(nn.Module):
    """SN-PatchGAN discriminator. Output is a map of real/fake scores."""

    def __init__(self, base: int = 64):
        super().__init__()
        c = base

        def block(cin, cout):
            return nn.Sequential(spectral_norm(nn.Conv2d(cin, cout, 5, 2, 2)),
                                 nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(
            block(4, c), block(c, 2 * c), block(2 * c, 4 * c),
            block(4 * c, 4 * c), block(4 * c, 4 * c),
            spectral_norm(nn.Conv2d(4 * c, 1, 5, 1, 2)),
        )

    def forward(self, image, mask):
        return self.net(torch.cat([image, mask], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
