import torch
import torch.nn as nn
import argparse

from mlp_mixer import MlpMixer
from mlp_mixer.data import load_data
from mlp_mixer.train import train
from mlp_mixer.hype_search import hyperparameter_tuning


# hyperparameters
num_blocks = 2
patch_size = 4
hidden_dim = 16
tokens_mlp_dim = 32
channels_mlp_dim = 32
batch_size = 8
epochs = 15
optimizer = 'SGD'
init_lr = 1e-2
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
                         num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim)

        train_loader, test_loader = load_data(batch_size)
        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, weight_decay=1e-3)
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=1e-3)

        train(None, model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs)
