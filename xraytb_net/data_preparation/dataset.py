import os
import torch
from torch.utils.data import Dataset
from .preprocessing import LungPreprocessor


class CXRDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = os.path.join(root, split)
        self.transforms = transforms
        self.preprocessor = LungPreprocessor()

        self.samples = []

        # Define label map
        label_map = {"Normal": 0, "OtherDisease": 1, "TB": 2}

        for cls, label in label_map.items():
            class_dir = os.path.join(self.root, cls)
            for img in os.listdir(class_dir):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Preprocessing
        image = self.preprocessor.preprocess(img_path)

        # Apply transforms
        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = torch.tensor(label, dtype=torch.long)  # CrossEntropy expects long
        return {"image": image, "label": label}
