import torch
import torch.nn as nn

from mlp_mixer import MlpMixer
from mlp_mixer.data import load_data
from mlp_mixer.train import train


# hyperparameters
num_classes = 10
num_blocks = 2
patch_size = 4
hidden_dim = 16
tokens_mlp_dim = 32
channels_mlp_dim = 32

input_channels = 3  # RGB
input_size = 32  # For CIFAR10

batch_size = 8
epochs = 15

if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.device('cuda')
        print('cuda')

    model = MlpMixer(input_channels, input_size, num_classes,
                     num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim)

    train_loader, test_loader = load_data(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-5)

    train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs)
