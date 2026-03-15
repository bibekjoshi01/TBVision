import pandas as pd
from data_preparation.dataset import CXRDataset
from torch.utils.data import DataLoader

# prepare a tiny CSV for smoke test
df = pd.DataFrame([
    {"image_path":"test_images/1.png", "split":"train", "label":0},
    {"image_path":"test_images/2.png", "split":"train", "label":1},
])
df.to_csv("data/labels_small.csv", index=False)

# load
df = pd.read_csv("data/labels_small.csv")
ds = CXRDataset(df=df, img_root="data/raw", split="train", size=512)
dl = DataLoader(ds, batch_size=2, num_workers=2)

batch = next(iter(dl))

print("batch image shape:", batch["image"].shape)
print("batch label:", batch["label"])
print("example meta:", batch["meta"])