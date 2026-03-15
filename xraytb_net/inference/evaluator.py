from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..data_preparation.dataset import CXRDataset
from ..data_preparation.transforms import get_val_transforms
from . import ClassificationService
from .metrics import evaluate_model


def build_loader(
    split: str,
    data_dir: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> DataLoader:
    """Create a DataLoader for the chosen split so it can be evaluated in production."""
    transforms = get_val_transforms(image_size)
    dataset = CXRDataset(root=data_dir, split=split, transforms=transforms)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available()),
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = "test",
    data_dir: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    mode: str = "ensemble",
    backbones: Optional[list] = None,
    dropout: float = 0.3,
    use_mc_dropout: bool = False,
) -> dict:
    """Return evaluation metrics for the checkpoint plus the desired split."""
    loader = build_loader(
        split=split,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    service = ClassificationService(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        mode=mode,
        backbones=backbones,
        dropout=dropout,
        use_mc_dropout=use_mc_dropout,
    )
    return evaluate_model(
        service.model,
        loader,
        device=service.device,
        num_classes=len(service.label_map),
    )
