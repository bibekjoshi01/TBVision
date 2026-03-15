import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(size=224):
    return A.Compose(
        [
            A.RandomResizedCrop(size, size, scale=(0.85, 1.0), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GridDistortion(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
    )


def get_val_transforms(size=224):
    return A.Compose(
        [A.Resize(size, size), A.Normalize(mean=0.5, std=0.5), ToTensorV2()]
    )
