import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DiT(nn.Module):
    sqrt_alphas_cumprod: torch.Tensor # for training
    sqrt_one_minus_alphas_cumprod: torch.Tensor # for training
    sqrt_recip_alphas: torch.Tensor # for inferencing
    posterior_variance: torch.Tensor # for inferencing
    betas: torch.Tensor
    alphas: torch.Tensor
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
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.device = device
        self.max_timesteps = max_timesteps
        self.prepare_dit_params(beta_start, beta_end, max_timesteps)

        self.vis_emb = PatchEmbedding(
            in_channels=img_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )
        self.seq_len = (img_size // patch_size) ** 2
        self.time_emb = nn.Embedding(max_timesteps, hidden_dim)
        self.lab_emb = MutiLabelMLP(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.decoder = PatchDecoder(
            in_channels=img_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            img_size=img_size,
        )

    def prepare_dit_params(self, beta_start, beta_end, timesteps):
        """Prepare parameters for adding noise at training and inference time."""
        # Parameters for adding noise at training
        betas = torch.linspace(beta_start, beta_end, timesteps).to(self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # noised_img = sqrt(alpha_cumprod)*original_img + sqrt(1-alpha_cumprod)*noise
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # Parameters for adding noise at inference
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.betas = betas
        self.alphas = alphas

    def step_denoise(self, noised_vision, timesteps, label):
        """Return predicted noise from (noise + vision, label)"""
        features = self.vis_emb(noised_vision) + self.time_emb(timesteps).unsqueeze(1)
        features = self.lab_emb(features, label) # [batch_size, seq_len, hidden_dim]

        features = self.transformer_blocks(features)
        features = self.decoder(features)
        return features


    def forward(self, noise, label, vision=None):
        """
        Add noise and denoise using diffusion transformer.
        :param noise: [batch_size, in_channels, height, width]
        :param label: [batch_size]
        :param vision: [batch_size, in_channels, height, width]
        :return: [batch_size, in_channels, height, width]
        """
        b = noise.size(0)
        if vision is not None:
            # train step
            batch_timesteps = torch.randint(0, self.max_timesteps, (b, )).to(self.device)
            sqrt_alpha = self.sqrt_alphas_cumprod[batch_timesteps][:, None, None, None].to(self.device)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[batch_timesteps][:, None, None, None].to(self.device)
            noised_vision = sqrt_alpha * vision + sqrt_one_minus_alpha * noise

            predicted_noise = self.step_denoise(noised_vision, batch_timesteps, label)
            loss = F.mse_loss(predicted_noise, noise)
            return {'loss': loss}

        # infer step
        vision = torch.randn(noise.shape).to(self.device)
        for timestep in range(self.max_timesteps - 1, -1, -1):
            batch_timesteps = torch.full((b, ), timestep, device=self.device, dtype=torch.long)
            model_output = self.step_denoise(vision, batch_timesteps, label)

            vision = self.sqrt_recip_alphas[timestep] * (vision - self.betas[timestep]
                     / self.sqrt_one_minus_alphas_cumprod[timestep] * model_output)
            # add noise
            if timestep > 0:
                noise = torch.randn_like(vision)
                vision = vision + torch.sqrt(self.posterior_variance[timestep]) * noise
        return vision


