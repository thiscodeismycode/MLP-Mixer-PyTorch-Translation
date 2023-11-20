import os
import torch
import torchvision
import torchvision.transforms as transforms


def load_data(batch_size, is_cifar_10=True):

    if os.path.isdir('./data') is False:
        print('Creating folder to store data')
        os.mkdir('./data')

    if is_cifar_10:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                        transforms.RandomAffine(degrees=10, scale=(0.5, 2.))])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,
                                                  drop_last=True)

    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        transforms.Resize((256, 256))])

        train_set = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   drop_last=True)

        test_set = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                                  drop_last=True)

    return train_loader, test_loader
