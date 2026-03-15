import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from models.classifier import TBClassifier
from training.losses import FocalLoss
from data_preparation.dataset import CXRDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("weights", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def evaluate(model, loader, device="cuda", threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_binary = (all_preds > threshold).astype(int)

    acc = accuracy_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds_binary, average="binary"
    )

    # Sensitivity / Specificity
    tn, fp, fn, tp = confusion_matrix(all_labels, preds_binary).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall  # same as TP / (TP + FN)

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train(backbone, epochs=15, batch_size=16, lr=3e-4):
    print(f"\nStarting training for backbone: {backbone}\n")
    model = TBClassifier(backbone=backbone).to(DEVICE)

    # Load datasets
    # ----------------------------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = CXRDataset(df="data/labels.csv", img_root="data/raw", split="train")
    val_dataset = CXRDataset(df="data/labels.csv", img_root="data/raw", split="val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Training Loop
    # ----------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = FocalLoss(alpha=0.75, gamma=2)
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_val_auc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            with autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), avg_loss=running_loss / (loop.n + 1))

        avg_epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Avg Training Loss: {avg_epoch_loss:.4f}")

        # Validation
        val_metrics = evaluate(model, val_loader, device=DEVICE)
        print(
            f"Val AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | Sensitivity: {val_metrics['sensitivity']:.4f} | "
            f"Specificity: {val_metrics['specificity']:.4f}"
        )

        # Scheduler step
        scheduler.step(val_metrics["auc"])

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(
                model.state_dict(),
                f"weights/{backbone}_best_auc_{best_val_auc:.4f}.pth",
            )
            print("✅ Best model saved!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="densenet121")

    args = parser.parse_args()

    train(args.backbone)
