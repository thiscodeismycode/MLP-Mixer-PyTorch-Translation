from torch import nn
from einops.layers.torch import Rearrange

"""
Original code written in FLAX/JAX
https://arxiv.org/abs/2105.01601
https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py 
"""


# Input shape: [batch, channel, height, width]
# Layer size: height*width
class MlpBlock(nn.Module):
    def __init__(self, input_features, mlp_dim, dropout):
        super().__init__()
        self.input_features: int = input_features
        self.mlp_dim: int = mlp_dim
        self.dropout: float = dropout
        # y = nn.Dense(output_dim)(input)
        # FLAX nn.Dense() is a linear transformation applied over the last dimension of the input.
        self.mlp = nn.Sequential(
            nn.Linear(input_features, mlp_dim),  # nn.Linear(input_dim, output_dim)(input)
            nn.GELU(),
            nn.Linear(mlp_dim, input_features),  # Back to the original shape
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, patch_num, tokens_mlp_dim, channels_mlp_dim, dropout):
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim
        self.dropout: float = dropout

        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Rearrange('b n c -> b c n'),  # batch, hidden_dim, patch_num
            MlpBlock(patch_num, tokens_mlp_dim, dropout),
            Rearrange('b c n -> b n c')  # batch, patch_num, hidden_dim
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MlpBlock(hidden_dim, channels_mlp_dim)
        )

    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class MlpMixer(nn.Module):
    def __init__(self, input_channels, input_size, num_classes, num_blocks, patch_size,
                 hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout):
        super().__init__()
        self.input_channels: int = input_channels
        self.input_size: int = input_size
        self.num_classes: int = num_classes
        self.num_blocks: int = num_blocks
        self.patch_size: int = patch_size
        self.hidden_dim: int = hidden_dim
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim
        self.dropout: float = dropout

        patch_num: int = (input_size // patch_size) * (input_size // patch_size)

        # FLAX nn.Conv(features, kernel_size, strides=1, padding='SAME')
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # Create patches of size 'hidden_dim'
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),  # h X w = number of patches, c = hidden_dim
        )
        self.mixer_block = nn.ModuleList([])
        for _ in range(num_blocks):
            self.mixer_block.append(MixerBlock(hidden_dim, patch_num, tokens_mlp_dim, channels_mlp_dim, dropout))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Flatten(),
            nn.AvgPool1d(kernel_size=patch_num, stride=patch_num),
            nn.Linear(hidden_dim, num_classes),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):

        # Per-patch Fully-connected
        x = self.stem(x)

        # N X (Mixer Layer)
        for mixer_block in self.mixer_block:
            x = mixer_block(x)

        # Global Average Pooling + Fully connected
        output = self.head(x)

        return output
