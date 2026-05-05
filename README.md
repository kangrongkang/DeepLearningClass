# Deep Learning for Multi-class Alzheimer's MRI Classification

ELC 5365 final project — Kang Rong (April 2026).

Build and compare three models on the public Kaggle Alzheimer's MRI multiclass dataset
(~44k images, 4 severity classes): a custom CNN baseline, MobileNetV2 (transfer),
and VGG16 (transfer). Evaluate with accuracy / precision / recall / F1 / confusion
matrices and add Grad-CAM explanations on the best model.

## Repository layout

```
Project_DeepLearningClass/
├── archive/combined_images/       # 44k MRI images (4 class subfolders)
├── notebooks/                     # 01..07 notebooks (run in this order)
├── src/                           # shared config + helpers used by every notebook
│   ├── config.py                  # paths, classes, seeds, hyperparameters
│   ├── io_utils.py                # dataset listing / tf.data builders / persistence
│   ├── plot_utils.py              # consistent matplotlib styling
│   └── nb_bootstrap.py            # adds project root to sys.path
└── outputs/
    ├── figures/                   # PNG figures
    ├── models/                    # best model weights (.keras)
    ├── predictions/               # per-model test preds + metrics JSON
    ├── reports/                   # per-step Markdown reports
    └── splits/                    # train / val / test CSVs
```

## Notebook execution order

1. `01_data_inspection.ipynb` — class counts, mode (gray/RGB), sizes, sample grid, corruption check.
2. `02_data_preprocessing.ipynb` — stratified 70/15/15 split saved as CSVs, augmentation policy.
3. `03_model_a_baseline_cnn.ipynb` — custom CNN baseline.
4. `04_model_b_mobilenetv2.ipynb` — MobileNetV2 (frozen → fine-tune).
5. `05_model_c_vgg16.ipynb` — VGG16 (frozen, optional fine-tune).
6. `06_model_comparison.ipynb` — head-to-head metrics table + side-by-side confusion matrices.
7. `07_grad_cam.ipynb` — Grad-CAM on the best model.

Each notebook also writes a Markdown report to `outputs/reports/`.

## Running locally

The project uses **Keras 3 with the PyTorch backend** so training runs on
native Windows GPU. The RTX 5080 (Blackwell, sm_120) is supported by PyTorch
2.10+ with the cu128 wheels.

One-time setup:

```powershell
# CUDA-enabled PyTorch (sm_120-capable wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Keras 3 standalone
pip install keras

# Lightweight deps
pip install numpy pandas matplotlib pillow scikit-learn nbformat tqdm tabulate jupyter
```

Every notebook that touches keras starts with:

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
```

so the backend is locked before any keras import.

`configure_gpu()` reports the active device — verify it shows `"device": "GPU"`
and lists `NVIDIA GeForce RTX 5080` before training. The notebooks auto-degrade
to a 2-epoch / 2% smoke schedule when no GPU is visible.

**Why Keras 3 + PyTorch instead of native Windows TF?** Google removed GPU
support from native Windows TF after TF 2.10, and the old CUDA 11 wheels don't
support Blackwell at all. PyTorch's Windows GPU support is mature, and Keras 3
is backend-agnostic — model architectures, the training loop, and the
saved-model `.keras` format are unchanged.

**Colab still works.** Set `KERAS_BACKEND=tensorflow` (or unset it; Colab
defaults to TF) and the same notebooks run there.

## Reproducibility

* Global seed (`SEED = 42`) seeded across `random`, `numpy`, and `tensorflow`.
* Splits are written to CSV once; every model uses the same split.
* `outputs/predictions/<model>_metrics.json` and `<model>_test_predictions.npz`
  let `06_model_comparison.ipynb` rebuild the comparison table without retraining.

## Key dataset caveat

The Kaggle dataset is **augmented and upsampled**. Splitting at the image level
(rather than at the patient/scan level) can let very similar images leak across
splits and inflate test accuracy. Treat all reported numbers as a benchmark
comparison between architectures, not a clinical accuracy estimate. This caveat
is repeated in every notebook and in the final report.
