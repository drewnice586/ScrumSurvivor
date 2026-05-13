"""Wav2Lip model architecture — vendored from https://github.com/Rudrabha/Wav2Lip

Original paper:
    "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild"
    Prajwal et al., ACM MM 2020.

License: MIT (see https://github.com/Rudrabha/Wav2Lip/blob/master/LICENSE)

This file contains only the inference-path model code (no training components).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv2d(channels, channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class Wav2Lip(nn.Module):
    """Wav2Lip inference model.

    Inputs:
        audio_sequences: ``(B, 1, 80, T)`` — mel-spectrogram (T=16 frames).
        face_sequences:  ``(B, 6, 96, 96)`` — masked lower face + reference face.

    Output:
        ``(B, 3, 96, 96)`` — lip-synced face, pixel values in [0, 1] (Sigmoid).
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Audio encoder ────────────────────────────────────────────────────
        self.audio_encoder = nn.Sequential(
            conv2d(1, 32, 3, 1),
            conv2d(32, 32, 3, 1),
            conv2d(32, 64, 3, 1),
            conv2d(64, 64, 3, 1),
            conv2d(64, 128, 3, 1),
            conv2d(128, 256, 3, 1),
            conv2d(256, 512, 3, 1),
            conv2d(512, 512, 3, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        # ── Face encoder ─────────────────────────────────────────────────────
        self.face_encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(conv2d(6, 16, 7, 1)),           # 96×96
                nn.Sequential(conv2d(16, 32, 3, 2), conv2d(32, 32, 3, 1)),   # 48
                nn.Sequential(conv2d(32, 64, 3, 2), conv2d(64, 64, 3, 1)),   # 24
                nn.Sequential(conv2d(64, 128, 3, 2), conv2d(128, 128, 3, 1)),  # 12
                nn.Sequential(conv2d(128, 256, 3, 2), conv2d(256, 256, 3, 1)),  # 6
                nn.Sequential(conv2d(256, 512, 3, 2), conv2d(512, 512, 3, 1)),  # 3
                nn.Sequential(conv2d(512, 512, 3, 1), conv2d(512, 512, 1, 1)),  # 3
            ]
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.face_decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(conv2d(512, 512, 1, 1)),
                nn.Sequential(conv2d(1024, 512, 3, 1), conv2d(512, 512, 3, 1)),
                nn.Sequential(conv2d(1024, 256, 3, 1), conv2d(256, 256, 3, 1)),
                nn.Sequential(conv2d(512, 128, 3, 1), conv2d(128, 128, 3, 1)),
                nn.Sequential(conv2d(256, 64, 3, 1), conv2d(64, 64, 3, 1)),
                nn.Sequential(conv2d(128, 32, 3, 1), conv2d(32, 32, 3, 1)),
                nn.Sequential(conv2d(64, 16, 3, 1), conv2d(16, 16, 3, 1)),
            ]
        )

        self.output_block = nn.Sequential(
            conv2d(16 + 6, 8, 3, 1),
            nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(
        self, audio_sequences: torch.Tensor, face_sequences: torch.Tensor
    ) -> torch.Tensor:
        # Encode audio
        audio_embedding = self.audio_encoder(audio_sequences)  # (B, 512, 1, 1)

        # Encode face
        feats: list[torch.Tensor] = []
        x = face_sequences
        for encoder_block in self.face_encoder_blocks:
            x = encoder_block(x)
            feats.append(x)

        # Fuse audio into the bottleneck
        audio_emb = audio_embedding.expand_as(feats[-1])
        x = torch.cat([feats[-1], audio_emb], dim=1)

        # Decode with skip connections
        for i, decoder_block in enumerate(self.face_decoder_blocks):
            x = decoder_block(x)
            if i < len(feats) - 1:
                x = F.interpolate(
                    x, size=feats[-2 - i].shape[2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, feats[-2 - i]], dim=1)

        x = torch.cat([x, face_sequences], dim=1)
        return self.output_block(x)
