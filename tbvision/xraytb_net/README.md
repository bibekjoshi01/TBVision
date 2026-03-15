# xraytb_net

xraytb_net holds the chest x-ray preprocessing, backbone models, evaluation helpers, and entrpoint scripts for the three-class (Normal / OtherDisease / TB) classifier.

## Quick start

1. Create and activate your Python 3.12+ virtual environment.
2. Install the dependencies:
   ```bash
   pip install -r tbvision/xraytb_net/requirements.txt
   ```
3. Make sure the `weights/` directory contains a trained checkpoint (e.g., `weights/xraytb_net.pth`). Training writes best checkpoints to that path by default.

## Training

Train via the CLI entry point:

```bash
python -m tbvision.xraytb_net.training.train_classifier \
  --data-dir dataset \
  --save-dir weights \
  --epochs 40 \
  --batch-size 32
```

Key flags:

- `--mode {single,ensemble}`: whether to use a single backbone or ensemble of DenseNet121 / EfficientNet-B3 / ResNet50.
- `--backbones` / `--backbone`: specify organization for ensemble or single-run mode.
- `--mc-dropout`: keep dropout active at inference for uncertainty sampling.
- `--lr`, `--weight-decay`, `--dropout`, `--gamma`: control optimizer / regularization.

Training saves the best checkpoint and history in `weights/` and logs metrics (accuracy, AUC, F1, class sensitivities, etc.).

## Inference

You can import `ClassificationService` from `tbvision.xraytb_net.inference` directly:

```python
from tbvision.xraytb_net.inference import ClassificationService

service = ClassificationService(
    checkpoint_path="weights/xraytb_net.pth",
    image_size=224,
    mode="ensemble",
    use_mc_dropout=False,
)

result = service.predict("/path/to/cxr.png")
print(result["prediction"], result["probabilities"])
```

`ClassificationService` wraps preprocessing, augmentation, device placement, and returns prediction/probabilities/logits.

## Evaluation utilities

- `tbvision.xraytb_net.inference.evaluate_model` works against dataloaders built by `tbvision.xraytb_net.data_preparation.dataset.CXRDataset`.
- Use `tbvision.xraytb_net.data_preparation.transforms.get_train_transforms` and `get_val_transforms` to build the augmentations used during training/testing.

## Data preparation

Customize dataset splits (train/val/test) under `dataset/` and rely on `CXRDataset` to pick up classes `Normal`, `OtherDisease`, and `TB`. Each image passes through the `tbvision.xraytb_net.data_preparation.preprocessing.LungPreprocessor` pipeline (CLAHE, mask, border crop) before augmentations.

## Notes

- Always keep the same `image_size` between training and inference (default 224).
- When running inside another repo component (e.g., backend), refer to this package as `tbvision.xraytb_net`.
- Weights can be converted to ONNX/ONNX Runtime later; the dependency list already includes `onnxruntime` for that purpose.
