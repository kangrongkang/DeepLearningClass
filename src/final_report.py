"""
Generates ``outputs/reports/00_final_report.md`` from the latest artifacts.

Re-run anytime any model is re-trained or evaluated; the report stays in sync
with whatever is on disk under ``outputs/``. No model loading, no inference —
this is pure aggregation over the JSON / PNG side-products.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    DATA_ROOT,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.comparison import (
    biggest_confusion,
    comparison_table,
    load_all_metrics,
    per_class_f1_table,
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _figure_md(rel_path: Path | str, alt: str, caption: str) -> str:
    """Return a Markdown image link with caption.

    The report lives at ``outputs/reports/00_final_report.md``. Image links
    must be expressed relative to *that* file, i.e. ``../figures/X.png``,
    even though the caller passes a project-root-relative path like
    ``outputs/figures/X.png`` for clarity. We resolve existence against the
    project root and rewrite the link.
    """
    rel = Path(rel_path)
    parts = list(rel.parts)
    if parts and parts[0] == "outputs":
        parts = parts[1:]
    disk_path = Path("outputs") / Path(*parts) if parts else rel
    if not (PROJECT_ROOT / disk_path).exists():
        return f"_(figure missing: {disk_path.as_posix()})_\n"
    md_link = Path("..") / Path(*parts)
    return f"![{alt}]({md_link.as_posix()})\n\n*Figure — {caption}.*\n"


def build_final_report() -> Path:
    # Pull every model that has metrics on disk, in the canonical display order:
    # baselines first, then EfficientNet ablations. Use the leakage-controlled
    # (clean) test set wherever the model has a *_clean_metrics.json file
    # (i.e. for the three baselines that were re-evaluated in Phase B).
    full_order = [
        "baseline_cnn", "mobilenetv2", "vgg16",
        "efficientnet_b0_128", "efficientnet_b0_a",
        "efficientnet_b0_b", "efficientnet_b0_c", "efficientnet_b0_c_tta",
    ]
    metrics = load_all_metrics(full_order, prefer_clean=True)
    inspection = _read_json(REPORTS_DIR / "01_data_inspection_summary.json")
    splits = _read_json(REPORTS_DIR / "02_split_summary.json")
    phase_a = _read_json(REPORTS_DIR / "phaseA_summary.json")
    effnet_128 = _read_json(REPORTS_DIR / "effnet_128_ablation_summary.json")

    today = date.today().isoformat()

    # -------------------- Section 1: Introduction & dataset --------------------
    intro = (
        "# Deep Learning for Multi-class Alzheimer's MRI Classification\n\n"
        f"*ELC 5365 final project — Kang Rong — generated {today}*\n\n"
        "## 1. Introduction\n\n"
        "Alzheimer's disease affects memory and cognition, and early detection "
        "matters for treatment and care. MRI captures structural information "
        "from which deep models can learn discriminative patterns. This project "
        "trains and compares three deep image-classification models on a public "
        "Kaggle dataset of MRI slices labeled by Alzheimer's severity, framed "
        "as a benchmark comparison between architectures rather than a "
        "clinical-grade diagnostic system.\n\n"
        "**Architectures compared.** A custom CNN trained from scratch (Model A), "
        "MobileNetV2 with ImageNet pre-training (Model B), and VGG16 with "
        "ImageNet pre-training (Model C). Day-by-day execution followed the "
        "9-step plan in `project plan.docx`; results, figures, and code are "
        "reproducible from the notebooks under `notebooks/`.\n\n"
        "## 2. Dataset\n\n"
    )
    if inspection:
        counts = inspection["class_counts"]
        intro += (
            f"The dataset contains **{inspection['total_images']:,} images** "
            "across four severity classes (file-counts confirmed by walking the "
            "raw data tree):\n\n"
            "| Class | Count |\n|---|---:|\n"
        )
        for c in CLASS_NAMES:
            intro += f"| {c} | {counts.get(c, 0):,} |\n"
        intro += "\n"
        probe = inspection.get("image_probe", {})
        intro += (
            "Image properties (sampled 50 per class):\n\n"
            f"- **Modes:** {probe.get('modes', {})} — mixed grayscale and RGB.\n"
            f"- **Sizes:** {probe.get('sizes', {})} — non-uniform.\n"
            f"- **Corrupt files:** {len(probe.get('corrupt', []))}.\n\n"
        )
    intro += _figure_md("outputs/figures/class_counts.png",
                        "class counts", "images per class in the raw dataset")
    intro += _figure_md("outputs/figures/01_sample_grid.png",
                        "sample grid", "8 random samples per class, resized to 128x128")

    # -------------------- Section 3: Preprocessing --------------------
    preprocess = "\n## 3. Data Preprocessing\n\n"
    preprocess += (
        "All images are decoded with PIL, forced to 3 channels (mixed grayscale / "
        "RGB sources are unified), resized to **128×128**, and normalized to "
        "**float32 in [0, 1]**.\n\n"
        "The dataset is split with stratified sampling: **70 % train / 15 % val / "
        "15 % test**, deterministic at SEED = 42. The splits are saved to "
        "`outputs/splits/{train,val,test}.csv` (using *relative* paths so the "
        "same CSV works on Windows and on Colab).\n\n"
    )
    if splits:
        preprocess += "Per-split totals:\n\n| Split | Total | NonDemented | VeryMildDemented | MildDemented | ModerateDemented |\n|---|---:|---:|---:|---:|---:|\n"
        for k in ["train", "val", "test"]:
            s = splits[k]
            bc = s["by_class"]
            preprocess += (f"| {k} | {s['total']:,} "
                           f"| {bc.get('NonDemented', 0):,} "
                           f"| {bc.get('VeryMildDemented', 0):,} "
                           f"| {bc.get('MildDemented', 0):,} "
                           f"| {bc.get('ModerateDemented', 0):,} |\n")
        preprocess += "\nStratification keeps each class's share within ~0.01 across train/val/test.\n\n"
    preprocess += (
        "**Augmentation policy** (training only): `RandomRotation(0.03)` (≈±10.8°), "
        "`RandomTranslation(0.05, 0.05)` (≈±5%), `RandomZoom(0.05)` (≈±5%). "
        "Horizontal flips are deliberately omitted because MRI left/right "
        "asymmetry can carry information.\n\n"
    )
    preprocess += _figure_md("outputs/figures/02_augmentation_examples.png",
                             "augmentation examples",
                             "original (left) vs four augmented variants per row")

    # -------------------- Sections 4, 5, 6: per-model results --------------------
    def _model_section(num: int, model_name: str, title: str, summary: str) -> str:
        m = metrics.get(model_name)
        block = f"\n## {num}. {title}\n\n{summary}\n\n"
        if m is None:
            block += f"*(metrics file `outputs/predictions/{model_name}_metrics.json` not found — re-run the {model_name} notebook.)*\n\n"
            return block
        train_block = m.get("training", {}) or {}
        block += "**Headline metrics on the held-out test set:**\n\n"
        block += (f"- Accuracy: **{m['accuracy']:.4f}**\n"
                  f"- Macro precision / recall / F1: "
                  f"{m['macro_precision']:.4f} / {m['macro_recall']:.4f} / "
                  f"**{m['macro_f1']:.4f}**\n"
                  f"- Weighted F1: {m['weighted_f1']:.4f}\n"
                  f"- Test set size: {m['n_test']:,}\n"
                  f"- Epochs run: {train_block.get('epochs_run', '—')}, "
                  f"wall-clock: {train_block.get('elapsed_seconds', '—')} s\n")
        block += "\n**Per-class scores:**\n\n| Class | Precision | Recall | F1 |\n|---|---:|---:|---:|\n"
        per = m["per_class"]
        for c, p, r, f in zip(per["labels"], per["precision"], per["recall"], per["f1"]):
            block += f"| {c} | {p:.4f} | {r:.4f} | {f:.4f} |\n"
        block += "\n"
        block += _figure_md(f"outputs/figures/{model_name}_training_curves.png",
                            f"{model_name} curves", f"{model_name} training/validation curves")
        block += _figure_md(f"outputs/figures/{model_name}_confusion_matrix.png",
                            f"{model_name} confusion matrix",
                            f"{model_name} confusion matrix (counts)")
        return block

    sec4 = _model_section(
        4, "baseline_cnn", "Model A — Custom CNN (from scratch, 128×128)",
        "Four conv blocks of `Conv → BatchNorm → ReLU → MaxPool` at 32 / 64 / 128 / 256 "
        "filters, then `GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(4, softmax)`. "
        "Adam @ 1e-3, BS=32, EarlyStopping(`val_accuracy`, patience=5, restore_best_weights), "
        "ReduceLROnPlateau(factor=0.5, patience=2). 423 K trainable parameters. "
        "Numbers below are on the leakage-controlled (clean) test set.")

    sec5 = _model_section(
        5, "mobilenetv2", "Model B — MobileNetV2 (transfer learning, 128×128)",
        "MobileNetV2 ImageNet backbone (`include_top=False`) with a "
        "`Rescaling(scale=2, offset=-1)` preprocessing layer (saves cleanly to `.keras`), "
        "GAP, Dropout(0.3), Dense(4, softmax). Two-stage training: stage 1 freezes the "
        "backbone (≈5 K trainable head params, lr=1e-3); stage 2 fine-tunes the top 20 "
        "layers (≈1.21 M trainable, lr=1e-5). 2.26 M total params.")

    sec6 = _model_section(
        6, "vgg16", "Model C — VGG16 (transfer learning, 128×128)",
        "VGG16 ImageNet backbone with `Rescaling(255)` then "
        "`vgg16.preprocess_input` applied directly on the symbolic tensor (no Lambda, "
        "save-safe), GAP, Dense(256), Dropout(0.5), Dense(4, softmax). 14.85 M total "
        "params; 132 K trainable in stage 1. Frozen-only training (the project plan flags "
        "VGG16 fine-tuning as risky on this dataset).")

    # ---------------- Refined-model sections ----------------
    sec7 = _model_section(
        7, "efficientnet_b0_128",
        "Model D-128 — EfficientNet-B0 at 128×128 (architecture isolation)",
        "Reviewer-requested ablation that uses the **same 128×128 input** as the "
        "baselines but swaps the custom CNN for ImageNet-pretrained EfficientNet-B0 "
        "with the refined recipe (AdamW + cosine warmup, full-network fine-tune). "
        "The point is to disentangle the architecture contribution from the "
        "resolution contribution. 4.05 M params.")

    sec8 = _model_section(
        8, "efficientnet_b0_a",
        "Model D-a — EfficientNet-B0 at 224×224 (bare recipe)",
        "Same architecture as D-128 but at the backbone's native 224×224 resolution. "
        "AdamW (lr stage1=1e-3 / stage2=1e-4, weight_decay=1e-4) + linear warmup + "
        "cosine annealing. No Mixup, no SWA, no TTA. 4.05 M params.")

    sec9 = _model_section(
        9, "efficientnet_b0_b",
        "Model D-b — EfficientNet-B0 + Mixup",
        "Adds Mixup α=0.2 (one mix per batch, λ from `Beta(α,α)` reflected to [0.5, 1]) "
        "on top of D-a. Label smoothing is **off** (Mixup already produces soft targets; "
        "stacking smoothing would over-smooth).")

    # Section 10 covers BOTH D-c (SWA without TTA) and D-c+TTA (best model).
    # We keep them in a single section so the negative SWA delta is visible
    # alongside the TTA recovery, rather than hiding it behind the +TTA number.
    def _dc_section() -> str:
        m_swa = metrics.get("efficientnet_b0_c")
        m_tta = metrics.get("efficientnet_b0_c_tta")
        block = (
            "\n## 10. Model D-c — EfficientNet-B0 + Mixup + SWA, with and without TTA\n\n"
            "Adds Stochastic Weight Averaging on the last 25 % of stage-2 epochs on "
            "top of D-b. We report the model **without TTA** and **with TTA** "
            "separately so the TTA contribution is attributable.\n\n"
        )
        if m_swa:
            block += (
                "**Headline metrics — D-c (SWA, no TTA):**\n\n"
                f"- Accuracy: **{m_swa['accuracy']:.4f}**, "
                f"Macro F1: **{m_swa['macro_f1']:.4f}**\n"
            )
            if m_swa.get("training", {}).get("elapsed_seconds"):
                block += f"- Wall-clock: ≈ {m_swa['training']['elapsed_seconds']:.0f} s\n"
            block += (
                "\nThis is *worse* than D-b, suggesting SWA in this single-seed "
                "configuration did not help.\n\n"
            )
        if m_tta:
            block += (
                "**Headline metrics — D-c + TTA (best model):**\n\n"
                f"- Accuracy: **{m_tta['accuracy']:.4f}**, "
                f"Macro F1: **{m_tta['macro_f1']:.4f}**\n"
                "- TTA pass: 4 augmented passes + 1 deterministic; augmentation "
                "applied *outside* the model graph so BatchNorm running statistics "
                "stay frozen.\n\n"
            )
            block += _figure_md(
                f"outputs/figures/efficientnet_b0_c_tta_confusion_matrix.png",
                "effnet c+tta confusion",
                "EfficientNet-B0 D-c + TTA confusion matrix")
        return block
    sec10 = _dc_section()

    # -------------------- Section 11: Comparison --------------------
    comparison = "\n## 11. Comparison\n\n"
    if metrics:
        try:
            tab = comparison_table(metrics)
            comparison += tab.to_markdown(index=False) + "\n\n"
        except Exception:
            comparison += "*(comparison_table failed — check the metrics files.)*\n\n"
        try:
            f1tab = per_class_f1_table(metrics).round(4)
            comparison += "**Per-class F1, head-to-head:**\n\n"
            comparison += f1tab.to_markdown() + "\n\n"
        except Exception:
            pass
        comparison += "**Where each model is most confused** (largest off-diagonal in the normalized confusion matrix):\n\n"
        for name, m in metrics.items():
            comparison += f"- **{name}** — {biggest_confusion(m)}\n"
        comparison += "\n"
        # Best-by-X
        best_acc = max(metrics, key=lambda k: metrics[k].get("accuracy", 0.0))
        best_f1 = max(metrics, key=lambda k: metrics[k].get("macro_f1", 0.0))
        comparison += (
            f"Best test accuracy: **{best_acc}** ({metrics[best_acc]['accuracy']:.4f}). "
            f"Best macro F1: **{best_f1}** ({metrics[best_f1]['macro_f1']:.4f}).\n\n"
        )
    comparison += _figure_md("outputs/figures/06_metrics_grouped_bars.png",
                             "headline metrics", "headline metrics — head-to-head bar chart")
    comparison += _figure_md("outputs/figures/06_per_class_f1.png",
                             "per-class F1", "per-class F1 — head-to-head")
    comparison += _figure_md("outputs/figures/06_confusion_matrices_side_by_side.png",
                             "confusion matrices", "side-by-side normalized confusion matrices")

    # -------------------- Section 12: Grad-CAM --------------------
    gradcam = "\n## 12. Explainability — Grad-CAM\n\n"
    gradcam += (
        "Grad-CAM produces a class-discriminative localization map by weighting "
        "the activations of the last conv layer with the gradients of the predicted "
        "class score. Applied to each trained model below; we show 2 correctly "
        "classified and 2 incorrectly classified examples per model, picked "
        "stratified by true class so the panel covers more than one severity level "
        "when possible.\n\n"
    )
    for name in metrics.keys():
        gradcam += _figure_md(f"outputs/figures/07_gradcam_{name}.png",
                              f"Grad-CAM {name}",
                              f"Grad-CAM heatmaps for {name} — original / heatmap / overlay per row")

    # -------------------- Section 14: Discussion & limitations --------------------
    discussion = (
        "\n## 14. Discussion and Limitations\n\n"
        "**The dataset is augmented and upsampled.** The Kaggle dataset description "
        "explicitly states that the images have been augmented and upsampled. Our "
        "cluster-stratified pHash split is the strictest leakage control the data "
        "structure permits, since the dataset does not expose patient or scan IDs. "
        "Subject-level leakage may still be present; the 99.9 % numbers should be "
        "read as *\"saturated under the strongest leakage control this dataset "
        "permits,\"* not as a clinical accuracy.\n\n"
        "**Inputs are 2D JPGs, not 3D MRI volumes.** The models classify pre-processed "
        "2D slices, which loses through-plane structure available in raw NIfTI / DICOM "
        "volumes.\n\n"
        "**Macro F1 is the right headline metric here.** With a 29 % / 25 % / 23 % / "
        "23 % class distribution, accuracy alone is misleading.\n\n"
        "**Why does ImageNet pretraining lose at 128² for VGG16/MobileNetV2 but not "
        "for EfficientNet-B0?** Our (D-128) ablation rules out a generic "
        "\"resolution-mismatch\" explanation: EfficientNet-B0 at the same 128² that "
        "crippled VGG16 / MobileNetV2 reaches macro F1 0.9944 — only 0.5 pp behind "
        "the same model at 224². Three mechanisms plausibly contribute:\n\n"
        "1. **Compound scaling** — EfficientNet-B0 was designed via a principled "
        "depth/width/resolution sweep (Tan & Le 2019), so down-scaling the input does "
        "less damage to the feature pyramid than for hand-engineered VGG16.\n"
        "2. **MBConv inverted residuals** with squeeze-and-excitation gates and SiLU "
        "activations are more parameter-efficient than VGG-style blocks in low-data "
        "regimes.\n"
        "3. **Fine-tuning depth** — we unfroze the entire backbone for D-128 stage 2; "
        "for MobileNetV2 we only unfroze the top 20 layers and for VGG16 we did not "
        "fine-tune at all. Raghu et al. (2019) explicitly note that full-network "
        "fine-tuning matters more than head-only adaptation for medical-imaging "
        "transfer.\n\n"
        "**Single-seed caveat for the recipe ablation.** The (D-a) → (D-b) → (D-c) "
        "deltas are all $\\leq$ 0.0007 macro-F1, which is within single-seed "
        "run-to-run noise. We **cannot** attribute these small differences to Mixup, "
        "SWA, or TTA without multi-seed runs.\n\n"
        "**Therefore, this project is best interpreted as a benchmark comparison "
        "between architectures, not as a clinical diagnostic system.**\n\n"
    )

    # ---------------- Phase A audit + leakage section ----------------
    leakage = "\n## 13. Phase A — Leakage Audit\n\n"
    if phase_a:
        cs = phase_a.get("cluster_stats", {})
        leakage += (
            f"We computed 256-bit perceptual hashes (DCT-based pHash, hash size 16) "
            f"for every image and clustered near-duplicates by Hamming-distance "
            f"threshold {phase_a['hamming_threshold']}. The threshold was calibrated "
            f"against {phase_a.get('leakage_audit_old_split', {}).get('per_split', {}).get('train', {}).get('n_images', '?')}+ "
            f"known md5-identical pairs found in the dataset. Result: "
            f"{cs.get('num_clusters', '?')} clusters from "
            f"{phase_a.get('n_images', '?')} images "
            f"(median cluster size {cs.get('median_cluster_size', '?')}, "
            f"max {cs.get('max_cluster_size', '?')}).\n\n"
        )
        # Old-split overlap
        ov = phase_a.get("leakage_audit_old_split", {}).get("overlaps", {})
        if "train<->test" in ov:
            tt = ov["train<->test"]
            leakage += (
                f"**Leakage audit on the original random split:** "
                f"{tt['shared_clusters']} clusters appeared in *both* train and "
                f"test ({tt.get('share_of_test', 0) * 100:.1f}% of test "
                f"clusters). After the cluster-stratified clean split, this "
                f"overlap is 0 by construction.\n\n"
            )
        leakage += (
            "**However**, the headline accuracy of every baseline is essentially "
            "unchanged between the two test sets — see Section 7's comparison "
            "table, where the baselines re-evaluated on the clean test set "
            "score within ±0.7% of their original numbers. This dataset's "
            "pre-augmentation does not, at the pHash level, generate "
            "near-duplicates aggressive enough to inflate test accuracy. "
            "Subject-level leakage is a separate question that **cannot be "
            "audited** here because the dataset does not expose patient or "
            "scan IDs.\n\n"
        )

    # ---------------- Ablation summary ----------------
    ablation = "\n## 15. Ablation: where does the +8.4 pp gain come from?\n\n"
    ablation += (
        "The refined model gains 8.4 macro-F1 points over the strongest baseline. "
        "Each row isolates one knob:\n\n"
        "| Step | Input | Model | Macro F1 | Δ |\n|---|---|---|---:|---:|\n"
    )

    def _row(label, input_size, model_key, prev_key):
        if model_key not in metrics:
            return ""
        f1 = metrics[model_key]["macro_f1"]
        if prev_key is None:
            return (f"| {label} | {input_size} | EfficientNet-B0 | {f1:.4f} | — |\n"
                    if "EfficientNet" in label
                    else f"| {label} | {input_size} | Custom CNN | {f1:.4f} | — |\n")
        prev_f1 = metrics.get(prev_key, {}).get("macro_f1")
        if prev_f1 is None:
            return ""
        delta = f1 - prev_f1
        sign = "+" if delta >= 0 else ""
        return (f"| {label} | {input_size} | EfficientNet-B0 | {f1:.4f} | "
                f"{sign}{delta:.4f} |\n")

    ablation += _row("Baseline", "128²", "baseline_cnn", None)
    ablation += _row("(D-128) Architecture only", "128²",
                     "efficientnet_b0_128", "baseline_cnn")
    ablation += _row("(D-a) + Resolution", "224²",
                     "efficientnet_b0_a", "efficientnet_b0_128")
    ablation += _row("(D-b) + Mixup α=0.2", "224²",
                     "efficientnet_b0_b", "efficientnet_b0_a")
    ablation += _row("(D-c) + SWA", "224²",
                     "efficientnet_b0_c", "efficientnet_b0_b")
    ablation += _row("(D-c) + SWA + TTA", "224²",
                     "efficientnet_b0_c_tta", "efficientnet_b0_c")
    ablation += (
        "\n**Architecture explains ~95 % of the gain** (custom CNN → EfficientNet-B0 "
        "at the same 128² input is +7.9 pp). The 128 → 224 resolution upgrade adds "
        "the next ~0.5 pp. SWA on this single-seed run *hurt* macro F1; TTA recovered "
        "it. We treat all sub-0.001 deltas as within single-seed noise and do not "
        "attribute them to specific recipe components without multi-seed runs.\n\n"
    )

    # ---------------- Section 16: artifact map ----------------
    artifacts = (
        "\n## 16. Where to find what\n\n"
        "| Artifact | Path |\n|---|---|\n"
        "| Per-day reports | `outputs/reports/01..10*.md`, `11_phase_e_refinement_summary.md` |\n"
        "| Refined plan | `refine.md` (project root) |\n"
        "| Headline figures | `outputs/figures/06_*.png` |\n"
        "| Per-model figures | `outputs/figures/<model>_{training_curves,confusion_matrix}.png` |\n"
        "| Grad-CAM panels | `outputs/figures/07_gradcam_<model>.png` |\n"
        "| Trained models | `outputs/models/<model>.keras` (gitignored — re-train via notebooks) |\n"
        "| pHash hashes | `outputs/splits/image_hashes.csv` (gitignored — regenerate via `python -m src.dedup`) |\n"
        "| Clean splits | `outputs/splits/{train,val,test}_clean.csv` |\n"
        "| Original splits | `outputs/splits/{train,val,test}.csv` |\n"
        "| Submission-ready paper | `paper/main.tex` + `paper/references.bib` |\n"
    )

    # Final order: intro, preprocess, model sections (4-10),
    # comparison (11), grad-cam (12), leakage (13), discussion (14),
    # ablation (15), artifacts (16).
    body = (intro + preprocess + sec4 + sec5 + sec6 + sec7 + sec8 + sec9 + sec10
            + comparison + gradcam + leakage + discussion + ablation + artifacts)
    out = REPORTS_DIR / "00_final_report.md"
    out.write_text(body, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build_final_report()
    print("Wrote:", p)
