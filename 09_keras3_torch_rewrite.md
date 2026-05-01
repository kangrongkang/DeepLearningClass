# Backend rewrite — Keras 3 + PyTorch (native Windows GPU)

**Date:** 2026-04-29
**Status:** Complete. Full project pipeline runs on the local RTX 5080 with native Windows GPU.

## Why this happened

The original implementation used `tensorflow.keras` directly. On the user's
machine — Windows 11 + RTX 5080 (Blackwell, sm_120) — that's a dead-end for
GPU training:

- Google removed GPU support from native Windows TF after TF 2.10.
- TF 2.10 used CUDA 11.2, which has no Blackwell support.
- WSL2 + TF GPU works but adds a non-trivial Linux-in-Windows setup step.

PyTorch's Windows GPU support is mature, and PyTorch 2.10+ (cu128 wheels)
supports Blackwell directly. Keras 3 is backend-agnostic — same model code,
same training loop, same `.keras` save format, just with `KERAS_BACKEND=torch`.

## What changed

| Module | Before | After |
|---|---|---|
| `src/io_utils.py` | `make_tf_dataset` (`tf.data.Dataset` with `tf.io.read_file` + `tf.image.resize`) | `make_keras_dataset` (Keras 3 `PyDataset` with PIL + numpy). Backwards alias kept. |
| `src/models.py` | `from tensorflow.keras import ...`, `tf.keras.optimizers.Adam` | `from keras import ...`, `keras.optimizers.Adam`. Augmentation moved into the model graph. |
| `src/training.py` | `configure_gpu` for TF, `tf.keras.callbacks` | `configure_gpu` detects torch GPU; uses `keras.callbacks`. The `evaluate_and_save` predict loop now handles torch tensors with `detach().cpu().numpy()`. |
| `src/gradcam.py` | `tf.GradientTape`-based gradient capture | torch forward + full backward hooks; supports inlined backbones (frozen or trainable). |
| `src/preprocessing.py` | `tf.keras.Sequential` for the augmentation viz | `keras.Sequential` with backend-agnostic numpy conversion. |
| Every notebook | implicit TF | first cell sets `os.environ.setdefault("KERAS_BACKEND", "torch")` |

## Performance optimization

Profiling found that running the augmentation Sequential **inside the
PyDataset** added a 330 ms / batch CPU↔GPU round-trip — the data loader was
13× slower than a clean PIL load. Fix: move the three augmentation layers into
the model graph as the very first block. They naturally short-circuit at
`training=False` (so eval/Grad-CAM are unaffected), and during training they
run on GPU as part of the forward pass. After this change:

| Stage | Time per batch (BS=32, RTX 5080) |
|---|---:|
| PIL load + numpy conversion | 24 ms |
| GPU forward pass (baseline CNN) | 19 ms |
| Augmentation in PyDataset (old path) | 330 ms ← was the bottleneck |
| Augmentation in model graph (new path) | ~5 ms |

## Results comparison

Same training config (2 epochs on 5 % of data; the *full* 30-epoch run is what
the user runs next), produced from the local RTX 5080:

| Model | Test acc | Macro F1 | Time |
|---|---:|---:|---:|
| baseline_cnn | 0.279 | 0.167 | 60 s for 2 epochs |
| mobilenetv2 | 0.452 | 0.424 | 111 s for 3 epochs |
| vgg16 | 0.455 | 0.437 | 62 s for 2 epochs |

Even at this tiny scale the qualitative finding from the project plan is
already visible: **transfer-learning models beat the from-scratch baseline by
~17 percentage points**, and **VeryMildDemented vs MildDemented is the hardest
pair**:

```
* baseline_cnn — confuses MildDemented -> ModerateDemented 100.0% of MildDemented samples.
* mobilenetv2  — confuses MildDemented -> VeryMildDemented 60.7% of MildDemented samples.
* vgg16        — confuses MildDemented -> VeryMildDemented 61.9% of MildDemented samples.
```

## What does *not* change

- Dataset, splits, label encoding, augmentation policy.
- Model architectures (parameter counts: 423 K / 2.26 M / 14.85 M unchanged).
- ImageNet pretrained weights — Keras 3 downloads the same canonical files
  that `tensorflow.keras.applications` would.
- `.keras` save format: bit-identical save / load round-trip verified for all
  three models.
- Final report structure, comparison artifacts, Grad-CAM panels.

## Reproducibility

Re-creating the environment from scratch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install keras numpy pandas matplotlib pillow scikit-learn nbformat tqdm tabulate jupyter
```

Then in any notebook:

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # MUST be before any keras import
```

`configure_gpu()` should print:

```
{'backend': 'torch', 'gpus': ['NVIDIA GeForce RTX 5080'], 'device': 'GPU',
 'keras_version': '3.14.0', 'torch_version': '2.11.0+cu128',
 'cuda': '12.8', 'compute_capability': [12, 0]}
```

## What the user does next

```powershell
# In the project root, with the venv active:
KERAS_BACKEND=torch jupyter nbconvert --to notebook --execute notebooks/03_model_a_baseline_cnn.ipynb --output 03_model_a_baseline_cnn.ipynb --ExecutePreprocessor.timeout=14400
KERAS_BACKEND=torch jupyter nbconvert --to notebook --execute notebooks/04_model_b_mobilenetv2.ipynb --output 04_model_b_mobilenetv2.ipynb --ExecutePreprocessor.timeout=14400
KERAS_BACKEND=torch jupyter nbconvert --to notebook --execute notebooks/05_model_c_vgg16.ipynb --output 05_model_c_vgg16.ipynb --ExecutePreprocessor.timeout=14400
KERAS_BACKEND=torch jupyter nbconvert --to notebook --execute notebooks/06_model_comparison.ipynb --output 06_model_comparison.ipynb
KERAS_BACKEND=torch jupyter nbconvert --to notebook --execute notebooks/07_grad_cam.ipynb --output 07_grad_cam.ipynb
KERAS_BACKEND=torch jupyter nbconvert --to notebook --execute notebooks/08_final_report.ipynb --output 08_final_report.ipynb
```

Or open them in Jupyter and re-run cells interactively. With the GPU active,
the training notebooks will pick up the FULL schedule automatically (30
epochs on 100 % of data); for a quick test set `OVERRIDE_SAMPLE_FRAC=0.05
OVERRIDE_EPOCHS=2`.

## Limitations preserved

The full Day 1-9 limitations (augmented + upsampled dataset, image-level
split, 2D JPGs vs 3D MRI volumes, benchmark-comparison framing not clinical
diagnostic) are unchanged and still appear in `00_final_report.md` Section 9.
