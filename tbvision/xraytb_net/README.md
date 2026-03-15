# xraytb_net

`tbvision.xraytb_net` contains the training, inference, and data utilities for the TB-Vision chest X-ray classifier. It is a lightweight PyTorch stack that can operate either as a single-backbone classifier or as an ensemble of DenseNet, EfficientNet, and ResNet models.

## Requirements

Install the package requirements inside `tbvision/xraytb_net/requirements.txt`.

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r tbvision/xraytb_net/requirements.txt
```

## File layout

- `data_preparation/`: dataset loader, preprocessing (`LungPreprocessor`), and
  albumentations transforms.
- `models/`: `TBClassifier` wraps a single backbone and adapts the input conv to a
  single channel; `TBEnsemble` averages logits from multiple classifiers and supports
  Monte-Carlo dropout uncertainty.
- `training/`: `train_classifier.py` orchestrates dataset loading, optimizer/scheduler,
  focal loss, and checkpoint/history saving.
- `inference/`: `ClassificationService` for runtime predictions, `evaluator` for
  evaluating checkpoints against a split, and shared metrics utilities.

## Dataset layout

The training script expects a directory structured like:

```
dataset/
  train/
    Normal/
    OtherDisease/
    TB/
  val/
    ...
  test/
    ...
```

Each class folder should contain standard image files (`.png`, `.jpg`, `.jpeg`). `CXRDataset` will preprocess every image through `LungPreprocessor`
(trim borders, apply lung mask, normalize intensity, CLAHE, resize) before applying albumentations transforms defined in `data_preparation/transforms.py`.

## Training

Use the `train_classifier.py` script to train a single model or ensemble:

```bash
python -m tbvision.xraytb_net.training.train_classifier \
  --data-dir dataset \
  --save-dir weights \
  --mode ensemble \
  --backbones densenet121 efficientnet_b3 resnet50 \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --image-size 224
```

- `--mode` selects `ensemble` (default) or `single`.
- `--backbones` controls the individual models in an ensemble.
- `--mc-dropout` enables stochastic forward passes for uncertainty.
- Checkpoints named `<mode>-<backbones>_best.pth` are written to `--save-dir`,
  alongside `<mode>-<backbones>_history.json` summarizing losses and validation metrics.

The trainer uses `AdamW`, `ReduceLROnPlateau` on `val AUC`, and `FocalLoss` with inverse frequency class weights. Best models are automatically saved whenever the monitored metric improves.

## Inference & evaluation

- `ClassificationService` (`xraytb_net.inference`) wraps checkpoint loading, preprocessing, and prediction. It returns raw logits, per-class probabilities, binary interpretation, and uncertainty (if MC dropout is enabled).

- `TBEnsemble.predict_with_uncertainty` can be toggled by passing `--mc-dropout` to the trainer and instantiating the service with `use_mc_dropout=True`.

- `evaluate_checkpoint` accepts a checkpoint, split (`train`/`val`/`test`), and data dir, then runs the loader/evaluator to produce metrics such as AUC, sensitivity, specificity, confusion matrix, and per-class accuracy (see `inference/metrics.py`).

## Notes

- Keep pretrained weights in `weights/` (default) and point inference scripts/`ClassifierService` at the desired checkpoint path.
- You can reuse `LungPreprocessor`/`transforms` for other downstream tasks to ensure consistent preprocessing.
