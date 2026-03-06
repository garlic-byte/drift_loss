import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=patch_size, padding=1)

    def forward(self, x):
        """
        Embedding vision feature to input of transformer.
        :param x: shape [batch_size, in_channels, height, width]
        :return: shape [batch_size, seq_len, hidden_dim]
        """
        x = self.conv(x) # [batch_size, hidden_dim, height // patch_size, width // patch_size]
        x = x.flatten(2) # [batch_size, hidden_dim, seq_len]
        return x.transpose(1, 2)

class PatchDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size, img_size):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.linear = nn.Linear(hidden_dim, in_channels * patch_size * patch_size)

    def forward(self, x):
        """
        Decoder vision from output of transformer.
        :param x: shape [batch_size, seq_len, hidden_dim]
        :return: shape [batch_size, in_channels, height, width]
        """
        b = x.size(0)
        x = self.linear(x) # [batch_size, seq_len, in_channels * patch_size * patch_size]
        x = x.reshape(
            b,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.in_channels,
            self.patch_size,
            self.patch_size
        ).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, self.in_channels, self.img_size, self.img_size)


class MutiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.linear = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, hidden_dim]
        :return: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.linear(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [batch_size, num_heads, seq_len, head_dim]
        attn = F.scaled_dot_product_attention(q, k, v)
        return attn.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)


class MutiLabelMLP(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        self.w = nn.Parameter(0.02 * torch.randn(num_classes, hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_classes, hidden_dim))

    def forward(self, x, label):
        """
        :param x: input tensor, shape [batch_size, seq_len, hidden_dim]
        :param label: input label , shape [batch_size]
        :return: output tensor, shape [batch_size, seq_len, hidden_dim]
        """
        select_w = self.w[label]
        select_b = self.b[label]
        return torch.bmm(x, select_w) + select_b.unsqueeze(1)


def product_add(features, product, add):
    return features * (1 + product.unsqueeze(1)) + add.unsqueeze(1)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # self.attention1 = MutiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.attention1 = Attention(dim=hidden_dim, num_heads=num_heads)
        # self.attention2 = MutiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 6)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, condition):
        """
        :param x: [batch_size, seq_len, hidden_dim]
        :param condition: [batch_size, hidden_dim]
        :return: [batch_size, seq_len, hidden_dim]
        """
        gate1, prod1, add1, gate2, prod2, add2 = self.linear1(condition).chunk(6, dim=1)
        x = x + gate1.unsqueeze(1) * (
            self.attention1(
                product_add(self.norm1(x), prod1, add1)
            )
        )
        x = x + gate2.unsqueeze(1) * (
            self.linear2(
                product_add(self.norm2(x), prod2, add2)
            )
        )
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FlowMatching(nn.Module):
    def __init__(
            self,
            img_size,
            img_channels: int = 1,
            patch_size: int = 4,
            hidden_dim: int = 1024,
            num_heads: int = 64,
            beta_start=0.0001,
            beta_end=0.02,
            max_timesteps=1000,
            num_classes=10,
            num_layers=6,
            device='cuda:0',
            denoise_step=4,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.device = device
        self.denoise_step = denoise_step
        self.max_timesteps = max_timesteps

        self.vis_emb = PatchEmbedding(
            in_channels=img_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )
        self.seq_len = (img_size // patch_size) ** 2
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, self.seq_len, hidden_dim))
        self.time_emb = nn.Embedding(max_timesteps, hidden_dim)
        self.lab_emb = nn.Embedding(num_classes+1, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio=4.0) for _ in range(num_layers)
        ])
        self.decoder = PatchDecoder(
            in_channels=img_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            img_size=img_size,
        )

    def step_denoise(self, noised_vision, timesteps, label):
        """Return predicted noise from (noise + vision, label)"""
        features = self.vis_emb(noised_vision) + self.pos_emb # [batch_size, seq_len, hidden_dim]
        condition = self.time_emb(timesteps) + self.lab_emb(label)
        for block in self.transformer_blocks:
            features = block(features, condition)
        features = self.decoder(features)
        return features


    def forward(self, noise, label, vision=None):
        """
        Add noise and denoise using flow matching.
        :param noise: [batch_size, in_channels, height, width]
        :param label: [batch_size]
        :param vision: [batch_size, in_channels, height, width]
        :return: [batch_size, in_channels, height, width]
        """
        b = noise.size(0)
        if vision is not None:
            # train step
            batch_timesteps = torch.randint(0, self.max_timesteps, (b, )).to(self.device)
            version_ratio = (batch_timesteps / self.max_timesteps).reshape(-1, 1, 1, 1)
            noised_vision = version_ratio * vision + (1 - version_ratio) * noise
            velocity = vision - noise

            predicted_velocity = self.step_denoise(noised_vision, batch_timesteps, label)
            loss = F.mse_loss(predicted_velocity, velocity)
            return {'loss': loss}

        # infer step
        noised_vision = torch.randn(noise.shape).to(self.device)
        dt = 1 / self.denoise_step
        for timestep in range(0, self.max_timesteps, self.max_timesteps // self.denoise_step):
            batch_timesteps = torch.full((b, ), timestep, device=self.device, dtype=torch.long)
            pred_velocity = self.step_denoise(noised_vision, batch_timesteps, label)

            noised_vision = noised_vision + dt * pred_velocity
        return noised_vision


