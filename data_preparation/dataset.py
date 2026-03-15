import os
from typing import Optional, Dict
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

# for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


# --- helper ops ---
def apply_clahe(img: np.ndarray, clipLimit=2.0, tileGridSize=(8, 8)) -> np.ndarray:
    """Apply CLAHE on a single-channel uint8 image."""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # convert to 3-channel BGR->RGB if single channel: replicate channels to keep interface consistent
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class CXRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_root: str,
        split: str = "train",
        size: int = 512,
        apply_clahe_flag: bool = True,
        transforms: Optional[A.Compose] = None,
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_root = img_root
        self.size = size
        self.apply_clahe_flag = apply_clahe_flag
        self.transforms = transforms or self.default_transforms()

    def default_transforms(self):
        return A.Compose(
            [
                A.Resize(self.size, self.size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets={},
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"])
        img = read_image_rgb(img_path)  # HxWx3 RGB uint8

        # apply CLAHE to grayscale and then compose back to 3 channels
        if self.apply_clahe_flag:
            clahe_img = apply_clahe(img)
            img = np.stack([clahe_img, clahe_img, clahe_img], axis=-1)

        # optional: convert to float here before augmentations
        augmented = self.transforms(image=img)
        image = augmented["image"]  # Tensor CxHxW

        label = torch.tensor(row["label"], dtype=torch.float32)

        meta: Dict = {}
        # include demographic fields if present (keep for later fusion)
        for field in ("age", "sex", "other_meta"):
            if field in row.index:
                meta[field] = row[field]

        # placeholder for lung mask (filled later by segmentation step)
        lung_mask = torch.zeros_like(image[0:1, :, :])  # 1xHxW

        return {
            "image": image,
            "label": label,
            "meta": meta,
            "lung_mask": lung_mask,
            "image_path": row["image_path"],
        }
