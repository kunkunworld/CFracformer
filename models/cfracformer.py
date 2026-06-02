import math
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolType(Enum):
    MAX = 0
    AVG = 1
    MIN = 2


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b].", stacklevel=2)

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def get_pool2d(pool_type, kernel_size=3, stride=1, padding=0):
    if pool_type == PoolType.MAX:
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    if pool_type == PoolType.AVG:
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    if pool_type == PoolType.MIN:
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    raise ValueError(f"Invalid pool type: {pool_type}")


class LocalHolder2D(nn.Module):
    def __init__(self, offset=3, pool_type=PoolType.MAX):
        super().__init__()
        self.offset = offset
        self.pools = nn.ModuleList(
            [get_pool2d(pool_type, kernel_size=2 * i + 1, stride=1, padding=i) for i in range(1, offset + 1)]
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("LocalHolder2D expects input with shape [B, C, H, W].")
        device = x.device
        _, _, h, w = x.shape
        eps = 1e-8

        local_window = torch.tensor([2 * i + 1 for i in range(1, self.offset + 1)], dtype=x.dtype, device=device)
        local_img = torch.cat([pool(x).unsqueeze(-1) for pool in self.pools], dim=-1).clamp(min=eps)

        x_mat = torch.log10(local_window / (h * w)).view(1, 1, 1, 1, self.offset)
        x_mat = torch.cat([x_mat, torch.ones_like(x_mat)], dim=-2)
        x_mat = x_mat.expand(1, h, w, 2, self.offset)
        y_mat = torch.log10(local_img)

        xtx = x_mat @ x_mat.transpose(-2, -1)
        xtx = xtx + torch.eye(2, dtype=x.dtype, device=device).view(1, 1, 1, 2, 2) * 1e-6
        coeff = torch.inverse(xtx) @ x_mat @ y_mat.permute(0, 2, 3, 4, 1)
        return coeff.transpose(-2, -1)[..., 0].permute(0, 3, 1, 2)


class SPSD2D(nn.Module):
    def __init__(self, min_alpha=0, max_alpha=9, n_alpha=9, offset=3, pool_type=PoolType.MAX):
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.n_alpha = n_alpha
        self.delta = (max_alpha - min_alpha) / n_alpha
        self.local_holder = LocalHolder2D(offset=offset, pool_type=pool_type)

    def forward(self, x):
        holder = self.local_holder(x)
        alpha = torch.arange(self.n_alpha, dtype=x.dtype, device=x.device) * self.delta + self.min_alpha
        condition = (holder >= alpha[:, None, None, None, None]) & (
            holder < (alpha + self.delta)[:, None, None, None, None]
        )
        spsd = torch.sum(torch.sum((x * condition) ** 2, dim=-1), dim=-1)
        spsd = torch.nan_to_num(spsd)
        return spsd.permute(1, 2, 0)


class PWVD(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.padding = (window_size - 1) // 2

    def forward(self, x):
        b, c, h, w = x.shape
        pad = self.padding
        device = x.device
        idx_i = torch.arange(pad, h + pad, device=device).view(-1, 1, 1, 1)
        idx_i = idx_i + torch.arange(self.window_size, device=device).view(1, 1, -1, 1) - pad
        idx_j = torch.arange(pad, w + pad, device=device).view(1, -1, 1, 1)
        idx_j = idx_j + torch.arange(self.window_size, device=device).view(1, 1, 1, -1) - pad

        x = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)
        x = x[:, :, idx_i, idx_j]
        x = x * x.flip([-2, -1])
        x = torch.abs(torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1)))
        return x.reshape(b, c, h, w, self.window_size, self.window_size)


class SPHS(nn.Module):
    def __init__(self, n_pwvd=5, min_alpha=0, max_alpha=9, n_alpha=9, offset=3, pool_type=PoolType.MAX):
        super().__init__()
        self.n_pwvd = n_pwvd
        self.pwvd = PWVD(n_pwvd)
        self.spsd = SPSD2D(min_alpha, max_alpha, n_alpha, offset, pool_type)

    def forward(self, x):
        b, c, h, w = x.shape
        spectrum = self.pwvd(x).reshape(b * c, h * w, self.n_pwvd, self.n_pwvd)
        spectrum = self.spsd(spectrum)
        return spectrum.reshape(b, c, h, w, -1)


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, padding=1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0, drop_path=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path)
        self.input_resolution = input_resolution

    def forward(self, x):
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, c)
        windows = self.attn(windows).view(-1, self.window_size, self.window_size, c)
        x = window_reverse(windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return shortcut + self.drop_path(x)


class HSA(nn.Module):
    def __init__(self, dim=64, input_resolution=16, window_size=8, shift_size=4, num_heads=8):
        super().__init__()
        self.wa = SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size=0)
        self.swa = SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size=shift_size)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.wa(x)
        x = self.swa(x)
        x = x + self.ffn(self.norm(x))
        return x.permute(0, 3, 1, 2)


class ImgEmbed4SPHS(nn.Module):
    def __init__(self, img_height=64, img_width=64, patch_size=4, in_chans=2, embed_dim=64):
        super().__init__()
        self.h = img_height // patch_size
        self.w = img_width // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class SPHSBlock(nn.Module):
    def __init__(self, dim, n_pwvd=5, min_alpha=0, max_alpha=9, n_alpha=9, offset=3, pool_type=PoolType.MAX):
        super().__init__()
        self.softplus = nn.Softplus()
        self.sphs = SPHS(n_pwvd, min_alpha, max_alpha, n_alpha, offset, pool_type)

    def forward(self, x):
        return torch.sum(self.sphs(self.softplus(x)), dim=-1)


class MaskAttention(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, groups=n_feat)

    def forward(self, x):
        return x + x * torch.sigmoid(self.depth_conv(x))


class ReconEncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, input_resolution=16):
        super().__init__()
        self.hsa = HSA(embed_dim, input_resolution=input_resolution)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, n, c = x.shape
        size = int(n**0.5)
        feat = x.transpose(1, 2).contiguous().view(b, c, size, size)
        attn = self.hsa(feat).flatten(2).transpose(1, 2)
        x = self.norm1(x + attn)
        return self.norm2(x + self.ffn(x))


class ReconDecoder(nn.Module):
    def __init__(self, img_size=64, patch_size=4, embed_dim=64, num_layers=6, hidden_dim=256, out_channels=2):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.layers = nn.ModuleList(
            [ReconEncoderLayer(embed_dim, hidden_dim, input_resolution=img_size // patch_size) for _ in range(num_layers)]
        )
        self.linear_projection = nn.Linear(embed_dim, patch_size * patch_size)
        self.deconv = nn.ConvTranspose2d(patch_size * patch_size, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        b, n, _ = x.shape
        size = int(self.num_patches**0.5)
        x = self.linear_projection(x)
        x = x.transpose(1, 2).contiguous().view(b, self.patch_size * self.patch_size, size, size)
        return self.deconv(x)


class FracReconNet(nn.Module):
    def __init__(self, img_size=64, patch_size=4, embed_dim=64, num_layers=6, hidden_dim=256, out_channels=2):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [ReconEncoderLayer(embed_dim, hidden_dim, input_resolution=img_size // patch_size) for _ in range(num_layers)]
        )
        self.decoder = ReconDecoder(img_size, patch_size, embed_dim, num_layers, hidden_dim, out_channels)

    def forward(self, x, guidance):
        if guidance.dim() == 4:
            guidance = guidance.flatten(2).transpose(1, 2)
        for layer in self.encoder_layers:
            guidance = layer(guidance)
        return self.decoder(guidance)


class CFracFormer(nn.Module):
    def __init__(
        self,
        img_height=64,
        img_width=64,
        patch_size=4,
        in_chans=2,
        embed_dim=64,
        num_classes=3,
        n_pwvd=5,
        min_alpha=0,
        max_alpha=9,
        n_alpha=9,
        offset=3,
        mixer_depth=2,
        mixer_head=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.h = img_height // patch_size
        self.w = img_width // patch_size
        self.num_patches = self.h * self.w
        self.embed = ImgEmbed4SPHS(img_height, img_width, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.h, self.w))
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.sphs_block = SPHSBlock(embed_dim, n_pwvd, min_alpha, max_alpha, n_alpha, offset)
        self.mask = MaskAttention(embed_dim)
        self.reconstruction = FracReconNet(img_height, patch_size, embed_dim, out_channels=in_chans)
        self.recon_channel_adapter = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=mixer_head,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.mixer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=mixer_depth, norm=nn.LayerNorm(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.embed(x) + self.pos_embed
        sphs = self.mask(self.sphs_block(x))
        recon = self.reconstruction(x, sphs)
        recon = F.adaptive_avg_pool2d(self.recon_channel_adapter(recon), output_size=sphs.shape[-2:])
        x = self.beta * sphs + self.alpha * recon
        x = x.flatten(2).transpose(1, 2)
        x = self.mixer_blocks(self.norm(x))
        return x.mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


SPHSNetHSAMask = CFracFormer
