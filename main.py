import torch
import torch.nn as nn
import argparse

from mlp_mixer import MlpMixer
from mlp_mixer.data import load_data
from mlp_mixer.train import train
from mlp_mixer.hype_search import hyperparameter_tuning
from mlp_mixer.cosine_annealing_scheduler import CosineAnnealingWarmUpRestarts


# hyperparameters based on Mixer-S
batch_size = 32         # batch_size = 512 in the paper, but TOO LARGE for CIFAR10 dataset
num_blocks = 8
patch_size = 4
hidden_dim = 32         # hidden_dim = 512 in the paper, but TOO LARGE for CIFAR10 dataset
tokens_mlp_dim = 64     # tokens_mlp_dim = 256 in the paper, but TOO LARGE for CIFAR10 dataset
channels_mlp_dim = 128   # channels_mlp_dim = 2048 in the paper, but TOO LARGE for CIFAR10 dataset
epochs = 10
optimizer = 'SGD'
init_lr = 0.08
dropout = 0.3
# NOT to change: for CIFAR10
num_classes = 10
input_channels = 3  # RGB
input_size = 32  # CIFAR10 image size

# Add parsers
parser = argparse.ArgumentParser(description='Evolution?', argument_default="no")
parser.add_argument('-e', '--evol', type=str, required=False, help='Type "yes" for evolution')

evol = parser.parse_args().evol


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.device('cuda')
        print('cuda')

    if evol == 'yes':
        hyperparameter_tuning('hyperparameter.yaml')

    else:
        model = MlpMixer(input_channels, input_size, num_classes,
                         num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout)

        train_loader, test_loader = load_data(batch_size)
        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs//2,
                                                                         T_mult=1, eta_min=5e-3)
        # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=epochs//2, T_mult=1, eta_max=init_lr, T_up=5, gamma=0.5)

        train(None, model, train_loader, test_loader, batch_size, loss_fn, optimizer, scheduler, epochs)
