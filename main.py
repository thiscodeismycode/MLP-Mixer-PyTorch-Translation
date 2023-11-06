import torch
import torchvision
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

import mlp_mixer

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
