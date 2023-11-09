"""

"""
import yaml
import torch
import torch.nn as nn

from mlp_mixer import MlpMixer
from mlp_mixer.train import train
from mlp_mixer.data import load_data

input_channels = 3  # RGB
input_size = 32  # CIFAR10 image size
num_classes = 10


# TODO
def hyperparameter_tuning(yaml_path):
    with open(yaml_path) as f:
        hypes = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    for n_i in range(len(hypes["num_blocks"])):
        num_blocks = int(hypes["num_blocks"][n_i])
        for p_i in range(len(hypes["patch_size"])):
            patch_size = int(hypes["patch_size"][p_i])
            for h_i in range(len(hypes["hidden_dim"])):
                hidden_dim = int(hypes["hidden_dim"][h_i])
                for t_i in range(len(hypes["tokens_mlp_dim"])):
                    tokens_mlp_dim = int(hypes["tokens_mlp_dim"][t_i])
                    channels_mlp_dim = tokens_mlp_dim
                    for b_i in range(len(hypes["batch_size"])):
                        batch_size = int(hypes["batch_size"][b_i])
                        for e_i in range(len(hypes["epochs"])):
                            epochs = int(hypes["epochs"][e_i])
                            for i_i in range(len(hypes["init_lr"])):
                                init_lr = float(hypes["init_lr"][i_i])

                                h = [num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim,
                                     batch_size, epochs, init_lr]

                                model = MlpMixer(input_channels, input_size, num_classes,
                                                 num_blocks, patch_size, hidden_dim, tokens_mlp_dim,
                                                 channels_mlp_dim)

                                train_loader, test_loader = load_data(batch_size)
                                loss_fn = nn.CrossEntropyLoss()

                                optim = torch.optim.SGD(model.parameters(), lr=init_lr, weight_decay=1e-3)
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=epochs,
                                                                                                 eta_min=1e-4)

                                train(h, model, train_loader, test_loader, batch_size, loss_fn, optim, scheduler, epochs)
