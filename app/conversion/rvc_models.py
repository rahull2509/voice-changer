"""PyTorch neural architecture for RVC (Retrieval-based Voice Conversion) v1/v2 Synthesizer.

Implements the SynthesizerTrn generator network used by RVC checkpoints,
optimized for CPU inference with PyTorch.
"""

import math
import logging
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

logger = logging.getLogger(__name__)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    """HiFi-GAN Residual Block Type 1."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    """HiFi-GAN neural vocoder waveform generator."""

    def __init__(
        self,
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for c in self.ups:
            remove_weight_norm(c)
        for r in self.resblocks:
            r.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class TextEncoder(nn.Module):
    """Encodes HuBERT hidden representations and quantized pitch embeddings."""

    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, f0=True, version="v2"):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.f0 = f0
        self.version = version

        in_channels = 768 if version == "v2" else 256
        self.emb_phone = nn.Linear(in_channels, hidden_channels)

        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)

        # Simplified projection layers for inference
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch=None, lengths=None):
        # phone: [B, T, D]
        x = self.emb_phone(phone)
        if self.f0 and pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = x.transpose(1, 2)  # [B, C, T]
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs


class SynthesizerTrn(nn.Module):
    """Complete RVC VITS/HiFi-GAN synthesis model for CPU inference."""

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels=0,
        sr=40000,
        f0=True,
        version="v2",
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.resblock = resblock
        self.sr = sr
        self.f0 = f0
        self.version = version

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=f0,
            version=version,
        )

        self.dec = Generator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        if spk_embed_dim > 0:
            self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    @torch.no_grad()
    def infer(self, phone, pitch=None, sid=None):
        """Inference forward pass to synthesize raw waveform from phone features and pitch."""
        x, m, logs = self.enc_p(phone, pitch)
        g = None
        if hasattr(self, "emb_g") and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        o = self.dec(m, g=g)
        return o

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
