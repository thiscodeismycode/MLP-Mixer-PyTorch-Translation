import torch
import torch.nn as nn
import argparse

from mlp_mixer import MlpMixer
from mlp_mixer.data import load_data
from mlp_mixer.train import train
from mlp_mixer.hype_search import hyperparameter_tuning
from mlp_mixer.cosine_annealing_scheduler import CosineAnnealingWarmUpRestarts


optimizer = 'SGD'
# NOT to change: for CIFAR10
"""num_classes = 100
input_channels = 3  # RGB
input_size = 32  # CIFAR10 image size
"""
# For ImageNet-1k
num_classes = 1000
input_channels = 3  # RGB
input_size = 256  # All ImageNet images resized to 256*256

# Add parsers
parser = argparse.ArgumentParser(prog='MLP Mixer pytorch translation', description='Provide options',
                                 epilog='All arguments are optional.')
parser.add_argument('-y', '--hype', choices=[0, 1], type=int, required=False, default=0, help='Execute hyperparameter search')
parser.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='Batch size')
parser.add_argument('-p', '--patch_size', type=int, required=False, default=4, help='Image patch size')
parser.add_argument('-n', '--num_blocks', type=int, required=False, default=8, help='Number of MLP blocks')
parser.add_argument('-d', '--hidden_dim', type=int, required=False, default=32, help='Hidden dimension')
parser.add_argument('-t', '--tokens_mlp_dim', type=int, required=False, default=64, help='Token mixer dimension')
parser.add_argument('-c', '--channels_mlp_dim', type=int, required=False, default=128, help='Channel mixer dimension')
parser.add_argument('-l', '--learning_rate', type=float, required=False, default=0.08, help='Learning rate')
parser.add_argument('-r', '--dropout', type=float, required=False, default=0.5, help='Dropout')
parser.add_argument('-e', '--epochs', type=int, required=False, default=50, help="Training epochs")
parser.add_argument('-da', '--dataset', type=int, required=False, default=0, help='1 to train on CIFAR100')

args = parser.parse_args()
hype = args.hype
batch_size = args.batch_size
patch_size = args.patch_size
num_blocks = args.num_blocks
hidden_dim = args.hidden_dim
tokens_mlp_dim = args.tokens_mlp_dim
channels_mlp_dim = args.channels_mlp_dim
init_lr = args.learning_rate
dropout = args.dropout
epochs = args.epochs
is_cifar_10 = True if args.dataset == 0 else False


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.device('cuda')
        print('cuda')

    if hype > 0:
        hyperparameter_tuning('hyperparameter.yaml')

    else:
        model = MlpMixer(input_channels, input_size, num_classes,
                         num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout)

        train_loader, test_loader = load_data(batch_size, is_cifar_10)
        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=1e-4)

        train(None, model, train_loader, test_loader, batch_size, loss_fn, optimizer, scheduler, epochs)
