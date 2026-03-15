import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


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


def evaluate_model(model: torch.nn.Module, loader, device="cuda", num_classes=3):
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

    if not all_probs:
        raise ValueError("Evaluation loader produced zero batches.")

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    try:
        truth_onehot = np.eye(num_classes)[all_labels]
        auc = float(
            roc_auc_score(truth_onehot, all_probs, average="macro", multi_class="ovr")
        )
    except ValueError:
        auc = float("nan")

    conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    sensitivity, specificity = _compute_sensitivity_specificity(conf_mat)

    class_accuracy = []
    for idx in range(num_classes):
        denom = conf_mat[idx].sum()
        class_accuracy.append(float(conf_mat[idx, idx]) / denom if denom > 0 else 0.0)

    return {
        "accuracy": accuracy,
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
