"""
Grad-CAM helper for Day 7.

Works against any of the three models in this project. The model graph is the
single source of truth — we pick the last Conv2D layer programmatically rather
than hard-coding a name, so it adapts to the inlined MobileNetV2 and the
embedded VGG16 layouts equally well.

Public API:
    * ``find_last_conv_layer(model)``
    * ``gradcam_heatmap(model, image, last_conv_layer_name=None, pred_index=None)``
    * ``overlay_heatmap(image, heatmap, alpha=0.4)``
    * ``run_gradcam_panel(model_name, n_correct=2, n_wrong=2)``  -> figure path
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, FIGURES_DIR, IMG_SIZE, MODELS_DIR, PREDICTIONS_DIR
from src.io_utils import load_split, get_logger
from src.plot_utils import save_fig


def find_last_conv_layer(model) -> str:
    """Return the name of the last Conv2D layer in ``model``.

    Searches recursively into submodels — handles both the embedded layout
    (where the backbone is one layer with sub-layers) and the inlined layout
    (where the backbone's conv layers are top-level layers of the parent).
    """
    import keras

    last = None

    def _walk(m):
        nonlocal last
        for layer in m.layers:
            if isinstance(layer, keras.layers.Conv2D):
                last = layer
            if hasattr(layer, "layers") and layer.layers:
                _walk(layer)

    _walk(model)
    if last is None:
        raise RuntimeError(f"Could not find a Conv2D layer in model '{model.name}'.")
    return last.name


def gradcam_heatmap(model, image: np.ndarray,
                    last_conv_layer_name: str | None = None,
                    pred_index: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute a Grad-CAM heatmap for one image (Keras 3 + PyTorch backend).

    Returns ``(heatmap, predicted_class, full_probas)`` where ``heatmap`` is a
    2-D float array in [0, 1] at the spatial resolution of the chosen conv layer.

    Implementation notes:
      * We register a forward hook on the last conv layer to capture its
        activation, and the same tensor's ``register_hook`` to capture the
        gradient that flows back through it.
      * The model is set to ``.eval()`` so dropout is disabled and BN uses
        running stats — the heatmap should not depend on dropout draws.
    """
    import keras
    import torch

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Place the model in eval mode so the heatmap is reproducible.
    if hasattr(model, "eval"):
        model.eval()
    device = next(model.parameters()).device

    captured = {}

    def fwd_hook(module, inputs, output):
        captured["activation"] = output

    def bwd_hook(module, grad_input, grad_output):
        # ``grad_output`` is a tuple of grads w.r.t. each output of this module.
        # We want d(class_score) / d(layer_output), i.e. the first element.
        captured["grad"] = grad_output[0]

    fh = last_conv_layer.register_forward_hook(fwd_hook)
    bh = last_conv_layer.register_full_backward_hook(bwd_hook)

    try:
        # Setting requires_grad=True on the input ensures the autograd graph
        # builds end-to-end even if the backbone parameters are frozen — so the
        # backward hook on a layer inside that backbone still fires.
        x = torch.tensor(image[None, ...], device=device, dtype=torch.float32,
                         requires_grad=True)
        preds = model(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if pred_index is None:
            pred_index = int(preds[0].argmax().item())
        class_score = preds[0, pred_index]

        model.zero_grad(set_to_none=True)
        class_score.backward(retain_graph=False)

        activation = captured["activation"][0]
        grad = captured.get("grad")
        if grad is None:
            raise RuntimeError(
                "Backward hook did not fire on the last conv layer. "
                "Check the model is on the torch backend and the layer name is correct."
            )
        grad = grad[0]

        # Keras 3 default data_format is 'channels_last' (H, W, C). Confirm:
        if keras.backend.image_data_format() == "channels_last":
            # activation, grad: (H, W, C)
            pooled = grad.mean(dim=(0, 1))               # (C,)
            heatmap = (activation * pooled).sum(dim=-1)  # (H, W)
        else:
            # (C, H, W)
            pooled = grad.mean(dim=(1, 2))               # (C,)
            heatmap = (activation * pooled[:, None, None]).sum(dim=0)  # (H, W)

        heatmap = torch.relu(heatmap)
        denom = heatmap.max().clamp(min=1e-8)
        heatmap = heatmap / denom

        return (heatmap.detach().cpu().numpy(),
                pred_index,
                preds[0].detach().cpu().numpy())
    finally:
        fh.remove()
        bh.remove()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, cmap_name: str = "jet") -> np.ndarray:
    """Resize ``heatmap`` to the image size and overlay using a jet colormap."""
    h_uint8 = np.uint8(255.0 * heatmap)
    h_resized = np.array(
        Image.fromarray(h_uint8).resize((image.shape[1], image.shape[0]),
                                        resample=Image.BILINEAR)
    ) / 255.0
    cmap = matplotlib.colormaps[cmap_name]   # cm.get_cmap is deprecated since MPL 3.7
    jet_colors = cmap(h_resized)[..., :3]    # (H, W, 3) in [0, 1]
    img_rgb = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    img_rgb = np.clip(img_rgb, 0, 1)
    overlay = (1 - alpha) * img_rgb + alpha * jet_colors
    return np.clip(overlay, 0, 1)


def _load_image(rel_or_abs_path: str | Path) -> np.ndarray:
    """Load + resize one image to the model input format ([0, 1] float32, 128x128x3)."""
    img = Image.open(rel_or_abs_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


def run_gradcam_panel(model_name: str = "mobilenetv2",
                       n_correct: int = 2, n_wrong: int = 2,
                       seed: int = 42,
                       save_name: str | None = None) -> Path:
    """Build a Grad-CAM figure for the requested model.

    Loads:
        * ``outputs/models/<model_name>.keras``  — the saved best weights.
        * ``outputs/predictions/<model_name>_test_predictions.npz`` — picks
          which test images were predicted correctly vs incorrectly.
        * ``outputs/splits/test.csv`` — the file paths.

    Output: ``outputs/figures/<save_name>.png``. The figure is a grid where each
    row is one example image with three columns: original / Grad-CAM heatmap /
    overlay. Per-image titles state predicted vs true label and confidence.
    """
    import tensorflow as tf

    log = get_logger("gradcam")
    save_name = save_name or f"07_gradcam_{model_name}"

    model_path = MODELS_DIR / f"{model_name}.keras"
    pred_path = PREDICTIONS_DIR / f"{model_name}_test_predictions.npz"
    if not model_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            f"Cannot run Grad-CAM for '{model_name}': missing "
            f"{model_path.name if not model_path.exists() else pred_path.name}. "
            "Run that model's training notebook first."
        )

    log.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(model_path)
    last_conv = find_last_conv_layer(model)
    log.info("Last conv layer: %s", last_conv)

    preds = np.load(pred_path, allow_pickle=True)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    y_proba = preds["y_proba"] if "y_proba" in preds.files else None
    relpaths = preds["relpaths"] if "relpaths" in preds.files else None

    if relpaths is not None and len(relpaths) == len(y_true):
        # Build a lightweight DataFrame with the same column names as load_split
        from src.config import DATA_ROOT
        import pandas as pd
        test_df = pd.DataFrame({
            "relpath": [str(r) for r in relpaths],
            "filepath": [str(DATA_ROOT / str(r)) for r in relpaths],
        }).reset_index(drop=True)
    else:
        # Fallback for predictions saved before relpaths were persisted
        log.warning(
            "predictions npz has no 'relpaths' field — falling back to test split CSV. "
            "This only works if the model was evaluated on the full test set."
        )
        test_df = load_split("test").reset_index(drop=True)
        if len(test_df) != len(y_true):
            raise ValueError(
                f"Test split size ({len(test_df)}) doesn't match predictions "
                f"length ({len(y_true)}). Re-run the model notebook with relpaths."
            )

    rng = np.random.default_rng(seed)
    correct_mask = y_true == y_pred
    correct_idxs = np.where(correct_mask)[0]
    wrong_idxs = np.where(~correct_mask)[0]
    log.info("Correct in test: %d   Wrong: %d", len(correct_idxs), len(wrong_idxs))

    def _stratified_pick(idxs: np.ndarray, k: int) -> list:
        """Pick ~k images from idxs trying to spread across true classes.

        Round-robin across the classes present in idxs; falls back to plain
        random sampling if all picks come from a single class.
        """
        if k <= 0 or len(idxs) == 0:
            return []
        by_cls: dict[int, list[int]] = {}
        for i in idxs:
            by_cls.setdefault(int(y_true[i]), []).append(int(i))
        # shuffle each per-class bucket deterministically
        for c in by_cls:
            rng.shuffle(by_cls[c])
        chosen: list[int] = []
        # round-robin
        classes = sorted(by_cls.keys())
        while len(chosen) < k:
            advanced = False
            for c in classes:
                if not by_cls[c]:
                    continue
                chosen.append(by_cls[c].pop())
                advanced = True
                if len(chosen) >= k:
                    break
            if not advanced:
                break
        return chosen

    pick_correct = _stratified_pick(correct_idxs, n_correct)
    pick_wrong = _stratified_pick(wrong_idxs, n_wrong)
    picks = list(pick_correct) + list(pick_wrong)
    if not picks:
        raise RuntimeError("Empty pick set — predictions array seems empty.")

    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, n_rows * 3))
    if n_rows == 1:
        axes = np.array([axes])

    for r, idx in enumerate(picks):
        path = test_df.loc[idx, "filepath"]
        true_idx = int(y_true[idx])
        pred_idx = int(y_pred[idx])
        confidence = float(y_proba[idx, pred_idx]) if y_proba is not None else None

        image = _load_image(path)
        heatmap, _, _ = gradcam_heatmap(model, image, last_conv_layer_name=last_conv,
                                        pred_index=pred_idx)
        overlay = overlay_heatmap(image, heatmap)

        is_correct = true_idx == pred_idx
        marker = "✓" if is_correct else "✗"
        confidence_str = f"  (p={confidence:.2f})" if confidence is not None else ""
        title = (f"{marker}  pred: {CLASS_NAMES[pred_idx]}{confidence_str}\n"
                 f"true:  {CLASS_NAMES[true_idx]}")

        axes[r, 0].imshow(image)
        axes[r, 0].set_title("original" if r == 0 else "")
        axes[r, 1].imshow(heatmap, cmap="jet")
        axes[r, 1].set_title("Grad-CAM" if r == 0 else "")
        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title("overlay" if r == 0 else "")
        axes[r, 0].set_ylabel(title, fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.suptitle(f"Grad-CAM — {model_name}  (last conv: {last_conv})", y=1.005)
    fig.tight_layout()
    out = save_fig(fig, save_name)
    plt.close(fig)
    log.info("Saved Grad-CAM figure: %s", out)
    return out
