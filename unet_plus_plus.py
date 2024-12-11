#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 06:58:33 2024

@author: dagi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block: two Conv2D layers + BatchNorm + ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class NestedBlock(nn.Module):
    """Nested block for U-Net++ with dense skip connections."""
    def __init__(self, in_channels, mid_channels, out_channels):
        super(NestedBlock, self).__init__()
        self.conv = ConvBlock(in_channels + mid_channels, out_channels)

    def forward(self, x, skip_connection):
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        # Encoder blocks
        self.enc1 = ConvBlock(in_channels, 4)
        self.enc2 = ConvBlock(4, 16)
        self.enc3 = ConvBlock(16, 32)
        self.enc4 = ConvBlock(32, 64)
        self.enc5 = ConvBlock(64, 128)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Decoder blocks with proper concatenation sizes
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = NestedBlock(64, 64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = NestedBlock(32, 32, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = NestedBlock(16, 16, 16)

        self.up1 = nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2)
        self.dec1 = NestedBlock(4, 4, 4)

        # Final layer
        self.final = nn.Conv2d(4, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        c5 = self.enc5(p4)
        # Decoder with nested connections
        u4 = self.up4(c5)
        d4 = self.dec4(u4, c4)
        
        u3 = self.up3(d4)
        d3 = self.dec3(u3, c3)

        u2 = self.up2(d3)
        d2 = self.dec2(u2, c2)

        u1 = self.up1(d2)
        d1 = self.dec1(u1, c1)
        # Final output
        out = self.final(d1)
        return out



# Print model summary
# print(model)

