import torch
import torch.nn as nn
from utils import get_drift_loss

class PatchEmbed(nn.Module):
    def __init__(self, img_channels, hidden_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(img_channels, hidden_dim, kernel_size=3, stride=patch_size, padding=1)

    def forward(self, x):
        """Split images into patches and project to hidden dimension."""
        x = self.conv(x) # [batch_size, hidden_dim, h / patch_size, w / patch_size]
        x = x.flatten(2) # [batch_size, hidden_dim, h' * w']
        return x.transpose(1, 2) # [batch_size, h' * w', hidden_dim]

class DecoderEmbed(nn.Module):
    def __init__(self, img_channels, hidden_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_num = img_size // patch_size
        self.img_channels = img_channels
        self.linear = nn.Linear(hidden_dim, img_channels * patch_size * patch_size)

    def forward(self, x):
        """Project hidden dimension to image dimension."""
        batch_size = x.size(0)
        x = self.linear(x) # [batch_size, h' * w', img_channels * patch_size * patch_size]
        x = x.view(batch_size, self.patch_num, self.patch_num, self.img_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(batch_size, self.img_channels, self.img_size, self.img_size)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int = 4,
        num_classes: int = 10,
        img_channels: int = 1,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        assert img_size % patch_size == 0, f"{img_size} is not a multiple of {patch_size}."

        self.vision_embed = PatchEmbed(
            img_channels=img_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )
        self.seq_len = (img_size // patch_size) ** 2
        self.position_embed = nn.Parameter(
            0.02 * torch.randn(1, self.seq_len, hidden_dim),
        )
        self.label_embed = nn.Embedding(self.num_classes, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_embed = DecoderEmbed(
            img_channels=img_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            img_size=img_size,
        )

    def forward(self, noise, label, vision=None):
        """Vision Transformer Encoder Decoder forward pass."""
        vis_features = self.vision_embed(noise)
        vis_features = vis_features + self.position_embed # [batch_size, seq_len, hidden_dim]

        lab_features = self.label_embed(label) # [batch_size, hidden_dim]
        hidden_features = vis_features + lab_features.unsqueeze(1)

        hidden_features = self.transform(hidden_features)
        x = self.decoder_embed(hidden_features)
        if vision is not None:
            loss = get_drift_loss(
                inputs=vision,
                outputs=x,
                labels=label,
                num_classes=self.num_classes,
            )
            return {'loss': loss}
        return x

if __name__ == "__main__":
    model = VIT(img_size=28)

    inp = torch.randn(100, 1, 28, 28)
    lab = torch.arange(10, dtype=torch.long).unsqueeze(1).repeat(1, 10).flatten()
    out = model(inp, lab)
    print(out.shape)
