import einops
import numpy as np
from torch import nn

"""
Original code written in FLAX/JAX
https://arxiv.org/abs/2105.01601
https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py 
"""


# blocks
class MlpBlock(nn.Module):
    def __init__(self, mlp_dim):
        super().__init__()
        self.mlp_dim: int = mlp_dim

    def forward(self, x):
        dim = x.shape()[-1]
        # y = nn.Dense(output_dim)(input)
        # FLAX nn.Dense() is a linear transformation applied over the last dimension of the input.
        y = nn.Linear(dim, self.mlp_dim)(x)  # nn.Linear(input_dim, output_dim)(input)
        y = nn.GELU()(y)
        # Back to the original shape
        y = nn.Linear(y.shape()[-1], x.shape()[-1])
        return y


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim
        self.token_mixing = MlpBlock(tokens_mlp_dim)
        self.channel_mixing = MlpBlock(channels_mlp_dim)

    def forward(self, x):
        # FLAX LayerNorm does not change input shape!
        y = nn.LayerNorm(x.shape())(x)
        np.swapaxes(y, 1, 2)
        y = self.token_mixing(y)
        np.swapaxes(y, 1, 2)
        x = x + y

        y = nn.LayerNorm(x.shape())(x)
        y = self.channel_mixing(y)
        x = x + y

        return x


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.num_classes: int = num_classes
        self.num_blocks: int = num_blocks
        self.patch_size: int = patch_size
        self.hidden_dim: int = hidden_dim
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim
        self.mixer_block = MixerBlock(tokens_mlp_dim, channels_mlp_dim)

    def forward(self, x):
        s = self.patch_size
        # FLAX nn.Conv(features, kernel_size, strides=1, padding='SAME')
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        x = nn.Conv2d(x.shape()[-1], self.hidden_dim, (s, s), stride=(s, s))(x)
        x = einops.rearrange(x, 'n h w c -> n (h w ) c')
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
        x = nn.LayerNorm(x.shape())(x)
        x = np.mean(x, axis=1)
        output_head = nn.Linear(x.shape()[-1], self.num_classes)
        nn.init.zeros_(output_head.weight.data)
        output = output_head(x)
        return output
