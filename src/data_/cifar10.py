from __future__ import annotations
from torch.utils.data import Subset
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_transforms(augment: bool = True):
    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, test_tf


def get_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
    subset_fraction: float = 1.0,
    subset_seed: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = get_transforms(augment)
    train_set = CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set = CIFAR10(root=root, train=False, download=True, transform=test_tf)

    # Optional subset for smaller training/eval budgets
    if subset_fraction is not None and subset_fraction < 1.0:
        # treat non-positive fractions as "use full dataset"
        if subset_fraction <= 0.0:
            subset_fraction = 1.0

        n_train = len(train_set)
        n_test = len(test_set)
        k_train = max(1, int(round(n_train * subset_fraction)))
        k_test = max(1, int(round(n_test * subset_fraction)))

        g = torch.Generator()
        if subset_seed is not None:
            g.manual_seed(int(subset_seed))

        perm_train = torch.randperm(n_train, generator=g)
        perm_test = torch.randperm(n_test, generator=g)

        train_indices = perm_train[:k_train].tolist()
        test_indices = perm_test[:k_test].tolist()

        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
