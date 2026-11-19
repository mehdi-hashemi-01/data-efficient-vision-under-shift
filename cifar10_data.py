"""Shared CIFAR-10 download and loading functions."""
from pathlib import Path
from torchvision.datasets import CIFAR10

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"


def load_data():
    train = CIFAR10(root=str(DATA), train=True, download=True)
    test = CIFAR10(root=str(DATA), train=False, download=True)
    return train, test


