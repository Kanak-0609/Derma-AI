"""
unet.py - DermaAI U-Net Architecture

Standard U-Net for binary lesion segmentation, built from scratch
(encoder-decoder with skip connections).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoder_upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feat in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        reversed_features = features[::-1]
        prev_channels = features[-1] * 2
        for feat in reversed_features:
            self.decoder_upconvs.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(feat * 2, feat))
            prev_channels = feat

        # Final 1x1 conv -> raw logits (no sigmoid here; use BCEWithLogitsLoss)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encoder_block, pool in zip(self.encoder_blocks, self.pools):
            x = encoder_block(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx, (upconv, decoder_block) in enumerate(zip(self.decoder_upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skip_connections[idx]

            # Handle potential size mismatch (odd input dimensions) via center-crop/pad
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = decoder_block(x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick sanity check
    model = UNet(in_channels=3, out_channels=1)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
