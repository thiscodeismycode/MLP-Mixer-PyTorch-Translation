import torch
import torch.nn as nn

from mlp_mixer import MlpMixer
from mlp_mixer.data import load_data
from mlp_mixer.train import train


# hyperparameters
num_classes = 10
num_blocks = 4
patch_size = 4
hidden_dim = 8
tokens_mlp_dim = 16
channels_mlp_dim = 16

batch_size = 16
epochs = 10


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.device('cuda')
        print('cuda')

    model = MlpMixer(num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim)

    train_loader, test_loader = load_data(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train(model, train_loader, test_loader, loss_fn, optimizer, epochs)
