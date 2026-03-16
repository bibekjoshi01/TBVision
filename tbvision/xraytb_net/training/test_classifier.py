import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tbvision.xraytb_net.data_preparation.dataset import CXRDataset
from tbvision.xraytb_net.data_preparation.transforms import get_val_transforms
from tbvision.xraytb_net.inference.metrics import evaluate_model
from tbvision.xraytb_net.models.classifier import TBClassifier
from tbvision.xraytb_net.models.ensemble import TBEnsemble

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMAGE_SIZE = 224


def build_model(mode: str, backbones, dropout: float, use_mc_dropout: bool):
    if mode == "ensemble":
        return TBEnsemble(
            backbones=backbones, dropout_rate=dropout, use_mc_dropout=use_mc_dropout
        )
    return TBClassifier(backbone=backbones[0], dropout=dropout)
    return TBClassifier(
        backbone=backbones[0], dropout=dropout, use_mc_dropout=use_mc_dropout
    )


def run_test(args):
    if args.mode == "ensemble":
        backbone_list = args.backbones or ["densenet121", "efficientnet_b3", "resnet50"]
        model_tag = f"ensemble-{'-'.join(backbone_list)}"
    else:
        backbone_list = [args.backbone]
        model_tag = f"single-{backbone_list[0]}"

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(args.save_dir) / f"{model_tag}_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(
        args.mode, backbone_list, args.dropout, use_mc_dropout=args.mc_dropout
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    test_dataset = CXRDataset(
        root=args.data_dir, split="test", transforms=get_val_transforms(args.image_size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    metrics = evaluate_model(
        model, test_loader, device=DEVICE, num_classes=model.num_classes
    )

    class_acc_str = ", ".join(f"{acc:.3f}" for acc in metrics["class_accuracy"])
    print("\nTest Results:")
    print(
        f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | Sensitivity: {metrics['sensitivity']:.4f} | "
        f"Specificity: {metrics['specificity']:.4f} | Class acc: [{class_acc_str}]"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate CXR TB classifier on test set")
    parser.add_argument("--mode", choices=["single", "ensemble"], default="ensemble")
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument(
        "--backbones", nargs="+", help="List of backbones used when mode=ensemble"
    )
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--save-dir", type=str, default="weights")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
