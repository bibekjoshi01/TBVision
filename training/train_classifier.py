import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data_preparation.dataset import CXRDataset
from data_preparation.transforms import get_train_transforms, get_val_transforms
from models.classifier import TBClassifier
from models.ensemble import TBEnsemble
from training.losses import FocalLoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMAGE_SIZE = 224


def _compute_sensitivity_specificity(conf_mat: np.ndarray):
    tp = np.diag(conf_mat)
    fn = conf_mat.sum(axis=1) - tp
    fp = conf_mat.sum(axis=0) - tp
    tn = conf_mat.sum() - (tp + fn + fp)

    sensitivities = [
        tp_i / (tp_i + fn_i) if tp_i + fn_i > 0 else 0.0 for tp_i, fn_i in zip(tp, fn)
    ]

    specificities = [
        tn_i / (tn_i + fp_i) if tn_i + fp_i > 0 else 0.0 for tn_i, fp_i in zip(tn, fp)
    ]

    return float(np.mean(sensitivities)), float(np.mean(specificities))


def evaluate(model, loader, device: str = DEVICE, num_classes: int = 3):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    auc = float("nan")
    try:
        truth_onehot = np.eye(num_classes)[all_labels]
        auc = float(
            roc_auc_score(truth_onehot, all_probs, average="macro", multi_class="ovr")
        )
    except ValueError:
        pass

    conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    sensitivity, specificity = _compute_sensitivity_specificity(conf_mat)

    class_accuracy = []
    for idx in range(num_classes):
        denom = conf_mat[idx].sum()
        if denom > 0:
            class_accuracy.append(float(conf_mat[idx, idx]) / float(denom))
        else:
            class_accuracy.append(0.0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "class_accuracy": class_accuracy,
        "confusion_matrix": conf_mat.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def _compute_class_weights(dataset, num_classes: int = 3):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset.samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = (total / (counts * num_classes)).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def _safely_extract_metrics(metrics):
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "class_accuracy": metrics["class_accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def build_model(mode: str, backbones, dropout: float, use_mc_dropout: bool):
    if mode == "ensemble":
        return TBEnsemble(
            backbones=backbones, dropout_rate=dropout, use_mc_dropout=use_mc_dropout
        )
    return TBClassifier(backbone=backbones[0], dropout=dropout)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "ensemble":
        backbone_list = args.backbones or ["densenet121", "efficientnet_b3", "resnet50"]
        model_tag = f"ensemble-{'-'.join(backbone_list)}"
    else:
        backbone_list = [args.backbone]
        model_tag = f"single-{backbone_list[0]}"

    print(f"Backbones: {backbone_list} (mode={args.mode})")

    model = build_model(
        args.mode, backbone_list, args.dropout, use_mc_dropout=args.mc_dropout
    ).to(DEVICE)

    print("\nLoading datasets...")
    train_dataset = CXRDataset(
        root=args.data_dir,
        split="train",
        transforms=get_train_transforms(args.image_size),
    )
    val_dataset = CXRDataset(
        root=args.data_dir, split="val", transforms=get_val_transforms(args.image_size)
    )
    test_dataset = CXRDataset(
        root=args.data_dir, split="test", transforms=get_val_transforms(args.image_size)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = _compute_class_weights(train_dataset, num_classes=model.num_classes)
    print(f"Class weights (inverse freq normalized): {class_weights.tolist()}")
    loss_fn = FocalLoss(alpha=class_weights, gamma=args.gamma)
    scaler = GradScaler() if DEVICE == "cuda" else None

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.patience, factor=args.factor
    )

    best_metric_value = -float("inf")
    best_metric_label = "auc"
    checkpoint_path = Path(args.save_dir) / f"{model_tag}_best.pth"

    history = {"train_loss": [], "val_metrics": []}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE).long()

            optimizer.zero_grad()

            with autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), avg_loss=running_loss / (loop.n + 1))

        avg_epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Avg Training Loss: {avg_epoch_loss:.4f}")

        val_metrics = evaluate(
            model, val_loader, device=DEVICE, num_classes=model.num_classes
        )
        class_acc_str = ", ".join(f"{acc:.3f}" for acc in val_metrics["class_accuracy"])
        print(
            f"Val AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | Sensitivity: {val_metrics['sensitivity']:.4f} | "
            f"Specificity: {val_metrics['specificity']:.4f} | Class acc: [{class_acc_str}]"
        )

        history["train_loss"].append(avg_epoch_loss)
        history["val_metrics"].append(
            {
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
                "auc": val_metrics["auc"],
                "sensitivity": val_metrics["sensitivity"],
                "specificity": val_metrics["specificity"],
                "class_accuracy": val_metrics["class_accuracy"],
            }
        )

        scheduler_metric = val_metrics["auc"]
        if np.isnan(scheduler_metric):
            scheduler_metric = val_metrics["accuracy"]
        scheduler.step(scheduler_metric)

        metric_value = val_metrics["auc"]
        metric_label = "auc"
        if np.isnan(metric_value):
            metric_value = val_metrics["accuracy"]
            metric_label = "accuracy"

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_metric_label = metric_label
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Best model saved based on {metric_label} = {metric_value:.4f}\n")

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    print("\nFinal evaluation on best model")
    best_val_metrics = evaluate(
        model, val_loader, device=DEVICE, num_classes=model.num_classes
    )
    final_test_metrics = evaluate(
        model, test_loader, device=DEVICE, num_classes=model.num_classes
    )

    print("\nValidation Results:")
    print(
        f"Val AUC: {best_val_metrics['auc']:.4f} | F1: {best_val_metrics['f1']:.4f} | "
        f"Acc: {best_val_metrics['accuracy']:.4f}"
    )

    print("\nTest Results:")
    print(
        f"Test AUC: {final_test_metrics['auc']:.4f} | F1: {final_test_metrics['f1']:.4f} | "
        f"Acc: {final_test_metrics['accuracy']:.4f}"
    )

    results = {
        "config": {
            "mode": args.mode,
            "backbones": backbone_list,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "gamma": args.gamma,
            "class_weights": class_weights.tolist(),
        },
        "best_val_metrics": _safely_extract_metrics(best_val_metrics),
        "test_metrics": _safely_extract_metrics(final_test_metrics),
        "history": history,
        "best_checkpoint": str(checkpoint_path),
        "best_metric": {
            "name": best_metric_label,
            "value": best_metric_value,
        },
    }

    history_path = Path(args.save_dir) / f"{model_tag}_history.json"
    with open(history_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n✅ Training complete. Best checkpoint: {checkpoint_path}")
    print(f"📊 Metrics + history saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CXR TB classifier")
    parser.add_argument("--mode", choices=["single", "ensemble"], default="ensemble")
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument(
        "--backbones", nargs="+", help="List of backbones used when mode=ensemble"
    )
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--save-dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
