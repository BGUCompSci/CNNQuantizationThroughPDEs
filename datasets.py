import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split

dataset_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet': 100}


def create_sets(train_set, test_set, batch_size: int, train_part: float, val_part: float, num_workers: int):
    assert train_part + val_part == 1, "train_part + val_part don't sum to one."
    lengths = [int(len(train_set) * train_part), int(len(train_set) * val_part)]
    train_set, val_set = random_split(train_set, lengths)
    train_loader = \
        torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validate_loader = \
        torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=num_workers)
    return train_loader, validate_loader, test_loader


def load_dataset(dataset_name: str, batch_size: int, train_part: float, val_part: float,
                 num_workers: int, test_only=False, data_root='./data'):
    if dataset_name in {'cifar10', 'cifar100'}:
        if dataset_name == 'cifar10':
            transform_mean = (0.4914, 0.4822, 0.4465)
            transform_std = (0.2023, 0.1994, 0.2010)
            dataset = torchvision.datasets.CIFAR10
        else:
            transform_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            transform_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            dataset = torchvision.datasets.CIFAR100

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std),
        ])
        train_loader, validate_loader, test_loader = create_sets(
            dataset(root=data_root, train=True, download=True, transform=transform_train),
            dataset(root=data_root, train=False, download=True, transform=transform_test),
            batch_size, train_part, val_part, num_workers)
    elif dataset_name in {'imagenet'}:
        """
        Source:
        https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        """
        train_dir = os.path.join(dataset_name, 'train/')
        val = '2012Val' if dataset_name == 'imagenet' else 'val'
        valdir = os.path.join(dataset_name, f'{val}/')

        train_transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        test_transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        if test_only:
            train_loader = None
        else:
            train_set = torchvision.datasets.ImageFolder(train_dir, train_transformer)
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True
            )

        test_set = torchvision.datasets.ImageFolder(valdir, test_transformer)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        validate_loader = test_loader  # for imageNet it's the same, test doesn't exist.

    else:
        raise NameError('Dataset not available')

    return train_loader, validate_loader, test_loader, dataset_classes[dataset_name]
