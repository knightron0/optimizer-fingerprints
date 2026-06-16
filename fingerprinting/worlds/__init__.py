from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass(frozen=True)
class WorldConfig:
    world_id: str = "cifar10_resnet18_v1"
    dataset: str = "cifar10"
    model: str = "resnet18_cifar"
    batch_size: int = 128
    seed: int = 0
    data_dir: Path = Path("data")
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_train_loader(config: WorldConfig, device: torch.device) -> DataLoader:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def build_world(config: WorldConfig, device: torch.device) -> tuple[nn.Module, DataLoader]:
    set_seed(config.seed)
    model = build_cifar_resnet18().to(device)
    loader = build_train_loader(config, device)
    return model, loader
