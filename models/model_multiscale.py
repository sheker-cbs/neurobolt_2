"""
--------------------------------------------------------
Multi-scale spectral encoder for EEG representation learning

This module is inspired by and builds upon the BIOT and TimesNet codebases.
We extend our gratitude to the authors for their contributions.
https://github.com/ycq091044/BIOT
https://github.com/thuml/Time-Series-Library
---------------------------------------------------------
"""

import math

import numpy as np
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer

from models.new_layers import BandAttention, ChannelAttention


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PatchWindowEmbedding(nn.Module):
    def __init__(self, emb_size=20, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, ts, fre_emb)
        out: (batch, ts_emb, fre_emb)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MSSEncoder(nn.Module):
    def __init__(
            self,
            emb_size=200,
            heads=8,
            depth=4,
            n_channels=26,
            scale1=100,
            input_length=3200,
            win_level=3,
            **kwargs
    ):
        super().__init__()

        self.scale1 = scale1
        self.input_length = input_length
        self.win_level = win_level
        self.patch_embedding = nn.ModuleList([
            PatchFrequencyEmbedding(
                emb_size=emb_size, n_freq=scale1 * 2 ** i // 2 + 1
            ) for i in range(win_level + 1)])

        self.ts_embedding = nn.ModuleList([
            PatchWindowEmbedding(
                emb_size=input_length // scale1,
                n_freq=input_length // (scale1 * (2 ** i))
            )
            for i in range(win_level + 1)
        ])
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        self.channel_tokens = nn.Embedding(n_channels, 200)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )
        self.band_attn = BandAttention(emb_size=200)
        self.channel_attn = ChannelAttention(emb_size=200)

    def stft(self, sample, n_fft):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=n_fft,
            hop_length=n_fft,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x, input_chans=None, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        band_weights_per_channel = []
        for i in range(x.shape[1]):
            channel_spec_emb_list = []
            for n in range(self.win_level + 1):
                channel_spec_emb_list.append(self.ts_embedding[n](
                    self.patch_embedding[n](
                        self.stft(x[:, i: i + 1, :], n_fft=self.scale1 * (2 ** n)))))

            channel_spec_emb, band_weights = self.band_attn(channel_spec_emb_list)
            band_weights_per_channel.append(band_weights)

            batch_size, ts, _ = channel_spec_emb.shape
            if input_chans is None:
                channel_token_emb = (
                    self.channel_tokens(self.index[i + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )
            else:
                channel_token_emb = (
                    self.channel_tokens(self.index[input_chans[i] + n_channel_offset])
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, ts, 1)
                )
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        emb = torch.cat(emb_seq, dim=1)
        n_channels = x.shape[1]
        ts_per_channel = ts
        emb, channel_weights = self.channel_attn(
            self.transformer(emb), n_channels=n_channels, ts_per_channel=ts_per_channel)
        self.last_channel_weights = channel_weights
        self.last_band_weights = torch.stack(band_weights_per_channel, dim=1)
        return emb


if __name__ == "__main__":
    x = torch.randn(16, 26, 3200)
    model = MSSEncoder(depth=4, heads=8)
    out = model(x)
    print(out.shape)
