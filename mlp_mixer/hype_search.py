import yaml
import torch
import torch.nn as nn

from mlp_mixer.mlp_mixer_pytorch import MlpMixer
from mlp_mixer.train import train
from mlp_mixer.data import load_data

input_channels = 3  # RGB
input_size = 32  # CIFAR10 image size
num_classes = 10

# Run for only 5 epochs and decide the best model
epochs = 5


def hyperparameter_tuning(yaml_path):
    print("Starting hyperparameter search!")

    with open(yaml_path) as f:
        hypes = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    best_option = {}
    best_loss = 1_000_000.

    for p_i in range(len(hypes["patch_size"])):
        patch_size = int(hypes["patch_size"][p_i])
        for n_i in range(len(hypes["num_blocks"])):
            num_blocks = int(hypes["num_blocks"][n_i])
            for h_i in range(len(hypes["hidden_dim"])):
                hidden_dim = int(hypes["hidden_dim"][h_i])
                for t_i in range(len(hypes["tokens_mlp_dim"])):
                    tokens_mlp_dim = int(hypes["tokens_mlp_dim"][t_i])
                    for c_i in range(len(hypes["channels_mlp_dim"])):
                        channels_mlp_dim = int(hypes["channels_mlp_dim"][c_i])
                        for b_i in range(len(hypes["batch_size"])):
                            batch_size = int(hypes["batch_size"][b_i])
                            for i_i in range(len(hypes["init_lr"])):
                                init_lr = float(hypes["init_lr"][i_i])

                                h = [num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim,
                                     batch_size, epochs, init_lr]

                                model = MlpMixer(input_channels, input_size, num_classes,
                                                 num_blocks, patch_size, hidden_dim, tokens_mlp_dim,
                                                 channels_mlp_dim, dropout=0.5)

                                train_loader, test_loader = load_data(batch_size)
                                loss_fn = nn.CrossEntropyLoss()

                                optim = torch.optim.SGD(model.parameters(), lr=init_lr, weight_decay=1e-3)
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=epochs,
                                                                                                 eta_min=1e-4)

                                loss = train(h, model, train_loader, test_loader, batch_size, loss_fn, optim,
                                             scheduler, epochs)
                                if loss < best_loss:
                                    best_loss = loss
                                    best_option["patch_size"] = int(hypes["patch_size"][p_i])
                                    best_option["num_blocks"] = int(hypes["num_blocks"][n_i])
                                    best_option["hidden_dim"] = int(hypes["hidden_dim"][h_i])
                                    best_option["tokens_mlp_dim"] = int(hypes["tokens_mlp_dim"][t_i])
                                    best_option["channels_mlp_dim"] = int(hypes["channels_mlp_dim"][c_i])
                                    best_option["batch_size"] = int(hypes["batch_size"][b_i])
                                    best_option["init_lr"] = float(hypes["init_lr"][i_i])

    print("End of hyperparameter tuning.")
    print("THE best option: ")
    for k, v in best_option.items():
        print(k, v)

    # Export best result as a .yaml file
    with open('hype_search.yaml', 'w') as f:
        yaml.dump(best_option, f, default_flow_style=True)
    f.close()

    return best_option
