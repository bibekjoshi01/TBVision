import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from data_preparation.segmentation_dataset import LungSegmentationDataset
from models.segmentation import LungUNet
from training.losses import BCEDiceLoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    dataset = LungSegmentationDataset(
        image_dir="data/segmentation/images",
        mask_dir="data/segmentation/masks",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )

    model = LungUNet().to(DEVICE)

    optimizer = Adam(model.parameters(), lr=1e-4)

    loss_fn = BCEDiceLoss()

    epochs = 20

    for _ in range(epochs):
        model.train()
        loop = tqdm(loader)

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), "lung_unet.pth")


if __name__ == "__main__":
    train()
