# refine.md — Refined Plan to Beat the 90.5% Baseline (with Honest Evaluation)

**Author:** project working notes — 2026-04-30
**Goal:** beat our `baseline_cnn` (test acc 0.9053, macro F1 0.9084) on the same dataset, on a **leakage-controlled** test set, using a model + recipe with peer-reviewed support.
**Hard constraint:** all comparisons must be on the *same test split*; the existing 3 trained models become baselines that the refined model has to beat.

---

## 0. Why this plan exists

Our completed run (full data, GPU, 30 / 20 / 15 epochs) gave us:

| Model | Test acc | Macro F1 | Wall-clock |
|---|---:|---:|---:|
| baseline_cnn (custom 423 K) | 0.9053 | 0.9084 | 3 h 14 min |
| mobilenetv2 (pretrained) | 0.7426 | 0.7430 | 1 h 51 min |
| vgg16 (pretrained) | 0.7290 | 0.7348 | 1 h 11 min |

The from-scratch CNN beat both ImageNet-pretrained backbones by ~17 points.
Two things make that suspicious:

1. **The Kaggle dataset is "augmented and upsampled" by the publisher.** The
   data folders mix UUID-named files (e.g. `00046ff7-...jpg`) with originals
   named `26 (19).jpg`, `mildDem97.jpg` — the parenthesized counter is the
   signature of a copy-with-augmentation pipeline. We have already found
   **60 byte-identical duplicates among 2,000 sampled ModerateDemented
   images** (within the same class). Perceptual-hash near-duplicates will
   add many more.
2. **Image-level random splitting** is the textbook recipe for
   late-augmentation leakage. Yagis et al. (2021, *Sci Rep* 11:22544)
   showed that on OASIS-derived data, slice-level vs subject-level splits
   inflate test accuracy by ~30 percentage points; a model trained with
   *random labels* still scored ~95% under slice-level splitting. Wen et
   al. (2020, MedIA, [arXiv:1904.07773](https://arxiv.org/abs/1904.07773))
   surveyed Alzheimer-CNN literature and found that more than half of
   papers had data leakage. Hernández et al. (2025,
   *Diagnostics* 15(18):2348) repeated this analysis on 2024-2025 papers
   and found the same pattern.

The TL;DR: our 90.5% is partly a measurement artifact, the transfer-learning
underperformance is partly a measurement artifact (the small high-capacity
custom CNN overfits leaked duplicates more than a frozen 2.26 M-param
backbone), and the *honest* gap between architectures is much smaller than
17 points. To know which model is actually better, we have to fix the
measurement first.

---

## 1. Findings from research (citation-backed)

### 1.1 Dataset lineage

The 44,000-image set is one of three Kaggle dumps with shared lineage; all
trace back to the 6,400-image preprocessed set
(`sachinkumar413/alzheimer-mri-dataset`,
[DOI 10.34740/kaggle/ds/2029496](https://commons.datacite.org/doi.org/10.34740/kaggle/ds/2029496))
which is widely believed to be derived from OASIS axial slices (CDR labels
0/0.5/1/2 match OASIS, not ADNI). The publisher applied affine augmentation
(rotation, zoom, translation, flip) **before** publishing the splits, so any
end-user random split admits leakage. The publisher does **not** expose
patient or scan IDs.

### 1.2 Reported accuracies in published papers

| Paper | Architecture | Reported acc | Split | Trustworthy? |
|---|---|---|---|---|
| Khan et al., *PLoS ONE* 2024, [link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0304995) | DenseNet-201 (TL) | 98.24% on AD5C | image-level | suspect leakage |
| ViTAD, *Brain Research* 2024, [link](https://www.sciencedirect.com/science/article/abs/pii/S0006899324005560) | ViT | 99.98% on this Kaggle | image-level | suspect leakage |
| EfficientNet-B2 + VGG16 ensemble, [PMC10093003](https://pmc.ncbi.nlm.nih.gov/articles/PMC10093003/) | ensemble | 97.35% | image-level | suspect leakage |

There is **no peer-reviewed paper** that we could find that reports
> 95% on this exact 44 K set under a *patient-level* split. Treat the
literature's headline numbers as upper bounds, not goals.

### 1.3 Why ImageNet transfer learning underperforms here

- **Raghu, Zhang, Kleinberg & Bengio (2019), "Transfusion: Understanding
  Transfer Learning for Medical Imaging"**, NeurIPS 2019,
  [arXiv:1902.07208](https://arxiv.org/abs/1902.07208). On retinal fundus
  and CheXpert tasks, ImageNet pretraining gave *negligible* gains over
  scratch training, and the high-level natural-image features (faces, dogs,
  cars) actively hurt small-data tasks because grayscale anatomy doesn't
  share that vocabulary. Lightweight from-scratch CNNs matched ResNet-50.
  This is exactly our situation.
- **Mei et al. (2022), "RadImageNet"**, *Radiology: AI*,
  [doi.org/10.1148/ryai.210315](https://pubs.rsna.org/doi/full/10.1148/ryai.210315).
  Pretraining on 1.35 M radiology images yields +4.0 – 9.4 % AUC over
  ImageNet on small medical datasets.
- **Practical contributors in our specific runs:** (a) we used 128 × 128
  inputs while ImageNet backbones expect 224 × 224 — this discards most of
  the pretraining benefit; (b) we only fine-tuned 20 of MobileNetV2's 154
  layers and didn't fine-tune VGG16 at all.

### 1.4 What works on small medical 2D classification

| Recipe item | Citation | Expected impact |
|---|---|---|
| Patient-level split / strict deduplication | Yagis et al. 2021 | reveals true accuracy (often -10 to -30 pts) |
| Resolution 128 → 224 | matches pretrained native | +2 – 5 pts |
| AdamW + cosine + linear warmup | Loshchilov & Hutter ICLR 2019, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | +0.5 – 2 pts |
| Mixup (α = 0.2) | Zhang et al. ICLR 2018, [arXiv:1710.09412](https://arxiv.org/abs/1710.09412) | +0.5 – 2 pts |
| Label smoothing (ε = 0.1) | Szegedy et al. CVPR 2016, [arXiv:1512.00567](https://arxiv.org/abs/1512.00567) | +0.3 – 1 pt |
| Focal loss (γ = 2) for class imbalance | Lin et al. ICCV 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | mostly recall on minority class |
| RandAugment / TrivialAugment | Cubuk et al. NeurIPS 2020, [arXiv:1909.13719](https://arxiv.org/abs/1909.13719); Müller & Hutter ICCV 2021, [arXiv:2103.10158](https://arxiv.org/abs/2103.10158) | +0.5 – 1.5 pts |
| Stochastic Weight Averaging | Izmailov et al. UAI 2018, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407) | +0.3 – 1 pt, ~free |
| Test-Time Augmentation (TTA) | Kim et al. *Inf Sci* 2023 | +0.3 – 1 pt at inference |
| Ensembling 3 architectures | classic | +0.5 – 1.5 pts |

### 1.5 Architectures, ranked

We pick architectures that combine modern inductive biases with a parameter
count that fits our ~6,000 effective training samples (after deduplication)
without overfitting.

| Rank | Model | Params | Native input | Why |
|---|---|---:|---:|---|
| **1** | **EfficientNet-B0** (ImageNet) | 5.3 M | 224×224 | Best params/accuracy ratio. Tan & Le 2019, [arXiv:1905.11946](https://arxiv.org/pdf/1905.11946). Reported 95-99% on this dataset (with leakage). |
| 2 | DenseNet-121 | 8.0 M | 224×224 | Dense connections help small-data tasks. Khan et al. PLoS ONE 2024 used DenseNet-201 → 98.24% (with leakage). |
| 3 | ConvNeXt-Tiny | 28 M | 224×224 | Modern CNN with transformer-era training tricks. Liu et al. 2022, [arXiv:2201.03545](https://arxiv.org/pdf/2201.03545). |

We will start with **EfficientNet-B0** as our primary refined model; if time
permits, add DenseNet-121 and ConvNeXt-Tiny as ensemble candidates.

---

## 2. Refined plan — 5 phases

### Phase A — Data audit and de-leakage *(no training; fast)*

A1. **Compute `imagehash.phash` (64-bit pHash) for every image** at
    `archive/combined_images`. Pin `imagehash==4.3.1` and `Pillow>=10,<12`
    in `requirements.txt` so PIL JPEG decoding stays stable. Store as
    `outputs/splits/image_hashes.parquet` (relpath, phash_hex, class).

A2. **Calibrate the Hamming threshold against the 60 known md5-duplicates**
    found in our quick audit. The threshold passes if it puts each
    md5-duplicate pair in the same cluster *and* doesn't create runaway
    mega-clusters (cluster size > 200). Default starting point: Hamming ≤ 8.
    Build a `cluster_id` column via union-find.

A3. Quantify leakage in our **current** splits (already on disk under
    `outputs/splits/{train,val,test}.csv`): how many cluster IDs appear in
    *both* train and test? Plot the histogram of cluster sizes by class.

A4. **Build the clean splits with cluster-level stratification by class.**
    For each class:
      - take all clusters whose *dominant* label is that class,
      - assign each cluster to one of {train, val, test} with proportions
        70 / 15 / 15 *by image count, not cluster count* (so big clusters
        don't blow the proportions),
      - shuffle deterministically with `numpy.random.default_rng(42)`.

    Write to `outputs/splits/{train,val,test}_clean.csv`. This guarantees
    near-duplicates of the same source slice never cross splits *and*
    preserves per-class proportions.

A5. **Sanity checks:**
      1. No cluster_id appears in more than one split.
      2. Test-set per-class counts within ±10 % of the original
         (NonDemented 29 % / VeryMild 25 % / Mild 23 % / Moderate 23 %).
      3. Cluster-size distribution per class — no class is dominated by a
         single mega-cluster.

Tools: `imagehash==4.3.1`, `pandas`, `numpy`. No GPU required.

### Phase B — Re-evaluate the 3 baselines on the clean test set *(no training; fast)*

B1. Load each existing checkpoint
    (`outputs/models/{baseline_cnn,mobilenetv2,vgg16}.keras`).

B2. Run inference on `test_clean.csv` **at each model's native resolution**
    (128 × 128 for all three baselines — they were trained at 128, the
    refined model is at 224, and we must not introduce a resolution mismatch
    here that masquerades as a leakage drop). The image-loading helper takes
    `target_size` as a parameter. Save predictions and metrics with
    `_clean` suffix:
    `outputs/predictions/<model>_clean_test_predictions.npz`,
    `outputs/predictions/<model>_clean_metrics.json`.

B3. Document the **drop** in accuracy / macro F1 vs the old leaky test
    set. The drop is a quantitative measure of how much leakage was inflating
    the headline numbers. (Expectation: large for baseline_cnn, smaller for
    the transfer models.)

> Critically, this turns the 90.5% baseline into a *real* number. Whatever
> that number is, the refined model has to beat it.

### Phase C — Implement the refined model + training recipe

C1. **New model factory:** `build_efficientnet_b0(num_classes=4, input_shape=(224, 224, 3), augment=True)`.
    - Uses `keras.applications.EfficientNetB0(include_top=False, weights="imagenet")`.
    - In-graph `Rescaling(scale=255.0)` → built-in `efficientnet.preprocess_input` directly on the symbolic tensor (no Lambda).
    - In-graph augmentation block: `RandomRotation(0.06)` + `RandomTranslation(0.05, 0.05)` + `RandomZoom(0.05)` + `RandomBrightness(0.1)` + `RandomContrast(0.1)`.
    - Head: GAP → Dropout(0.3) → Dense(4, softmax).
    - Two-stage training built into the notebook: head warm-up at lr=1e-3, then full fine-tune at lr=1e-5.

C2. **Optional secondary factories** (build but don't train unless time
    permits): `build_densenet121`, `build_convnext_tiny`. Same head and
    preprocessing pattern.

C3. **Refined training loop additions to `src/training.py`:**
    - `make_callbacks_v2`: AdamW optimizer (set on the *model* in the
      factory), `keras.callbacks.LearningRateScheduler` with linear warmup
      then cosine annealing, EarlyStopping (val macro F1, patience 5),
      ModelCheckpoint, optional **SWA in the last 25 % of epochs** via a
      custom callback.
    - **Mixup** as a `tf.data`-style transform inside the PyDataset
      (toggle `mixup_alpha=0.2`). Mixup produces soft targets, so the loss
      switches to `keras.losses.CategoricalCrossentropy(label_smoothing=0.0)`
      with one-hot labels — **we explicitly do not stack label smoothing
      on top of Mixup** because that double-smooths the targets.
    - **Label smoothing** is wired in but mutually exclusive with Mixup:
      `train_model(...., mixup_alpha=0.2, label_smoothing=0.0)` or
      `train_model(...., mixup_alpha=0.0, label_smoothing=0.1)`. Default:
      Mixup on, smoothing off (Mixup is the stronger regularizer).
    - **Focal loss** as an alternative loss function (Lin et al. 2017).
      Wired in but **off by default** — the dataset isn't strongly imbalanced.
    - **TTA at inference**: a thin wrapper that averages predictions over
      the original image and 4 small geometric perturbations.

C4. **Code-reviewer agent pass** on every new file before training kicks off.

### Phase D — Train on the clean split (3 attributable configs)

We deliberately split the refined-model training into three configurations
so any improvement is *attributable* to a specific recipe component, rather
than a single full-recipe run vs single bare-recipe run.

| Config | What's on | Expected to isolate |
|---|---|---|
| **D-a** Bare | EfficientNet-B0 + AdamW + cosine + warmup; no Mixup, no smoothing, no SWA | the architecture + resolution + optimizer baseline |
| **D-b** + Mixup | D-a + Mixup α = 0.2 (label smoothing OFF — see C3) | the Mixup contribution alone |
| **D-c** + SWA + TTA | D-b + SWA on last 25 % epochs + TTA at inference | regularization & inference-time gains |

D1. **Resolution: 224 × 224** (we keep the existing 128-input baselines
    untouched as historical baselines).

D2. **Per-config schedule** (same for D-a, D-b, D-c):
    - Stage 1 (frozen backbone): AdamW lr 1e-3, weight_decay 1e-4, 10 epochs,
      linear warmup 2 epochs then cosine.
    - Stage 2 (full fine-tune, all layers trainable): AdamW lr 1e-4,
      weight_decay 1e-4, 15 epochs, cosine.
    - SWA toggled on only in D-c (last 25 % ≈ 4 epochs of stage 2).

D3. **Save artifacts** with the same contract as Days 3-5, named by config:
    `outputs/models/efficientnet_b0_{a,b,c}.keras`,
    `outputs/predictions/efficientnet_b0_{a,b,c}_{metrics.json,test_predictions.npz}`,
    `outputs/figures/efficientnet_b0_{a,b,c}_{training_curves,confusion_matrix}.png`.
    All on the clean test set.

D4. **TTA pass** is part of config D-c: evaluate the same model with TTA
    active and record both `efficientnet_b0_c` (no TTA) and
    `efficientnet_b0_c_tta` so we can isolate the TTA contribution.

D5. **Time budget on RTX 5080 @ 224 × 224, BS = 64, ≈ 5–10 K train images,
    25 epochs total per config**: roughly 1–1.5 h per config →
    ~3–4.5 h total for all three configs. If we need to cut time, drop D-c
    first (SWA gives the smallest, most fragile gain).

### Phase E — Compare and report

E1. Build a single comparison table on the **clean test set**:

| Model | Source | Test acc | Macro F1 | Notes |
|---|---|---:|---:|---|
| baseline_cnn | from Days 3 (re-eval) | ? | ? | original 0.9053 was on leaky test |
| mobilenetv2 | from Day 4 (re-eval) | ? | ? | original 0.7426 |
| vgg16 | from Day 5 (re-eval) | ? | ? | original 0.7290 |
| **efficientnet_b0** | new | **?** | **?** | EfficientNet-B0 + recipe |
| efficientnet_b0 + TTA | new | ? | ? | with TTA |

E2. Write `outputs/reports/11_refinement_results.md` with the comparison
    table, per-class F1, confusion matrices, and a short discussion.
    Update `outputs/reports/00_final_report.md`.

E3. **Honest framing in the report:**
    - State the leakage hypothesis up front.
    - Report old (leaky) baseline AND new (clean) baseline.
    - Report refined-model gain over the *clean* baseline.
    - Cite the leakage references.

---

## 3. Expected impact (rough budget)

If our hypothesis is right, expected numbers on the clean test set:

| Model | Old (leaky) acc | Estimated clean acc | Source of estimate |
|---|---:|---:|---|
| baseline_cnn | 0.9053 | **0.70 – 0.82** | Yagis 2021 inflation magnitudes |
| mobilenetv2 | 0.7426 | 0.70 – 0.74 | small drop because backbone was frozen |
| vgg16 | 0.7290 | 0.70 – 0.73 | similar |
| **efficientnet_b0 + recipe** | n/a | **0.85 – 0.92** | resolution + recipe + better arch |

If `efficientnet_b0` lands ≥ 5 points above the clean `baseline_cnn`, we
have beaten the baseline with proof.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Perceptual hashing produces too-large clusters and we end up with a tiny clean dataset | Calibrate threshold against the 60 known md5-duplicates (Phase A2); report cluster-size distribution and adjust if max cluster > 200 |
| Stratifying by `cluster_id` skews the test class distribution | Stratify clusters by their *dominant class* and split per-class, weighted by image count (Phase A4). Sanity-check class fractions in Phase A5. |
| Phase D training time on RTX 5080 takes too long (224×224 is 3× the pixel count of 128) | Three configs × ~1–1.5 h each ≈ 3–4.5 h. Drop D-c (SWA + TTA) first if time-constrained. |
| EfficientNet-B0 + Mixup + smoothing changes too many things at once and we can't attribute gains | Three-config ablation D-a / D-b / D-c isolates each contribution. Mixup and label smoothing are explicitly mutually exclusive (see C3) to avoid double-soft targets. |
| Clean test set is too small after deduplication (e.g., < 1 k images) | Tighter Hamming threshold (≤ 6 or ≤ 4) and re-run; report the size each time. We accept smaller test sets in exchange for honesty. |
| The refined model still doesn't beat baseline_cnn on clean test | This is itself a publishable, honest result. Document it as such — it would mean either (a) the dataset is fundamentally a from-scratch-CNN-friendly task at the available resolution, or (b) our recipe still has gaps. We do *not* hide negative results. |

---

## 5. Bibliography

Foundational architecture & training:
- Tan, M. & Le, Q., **"EfficientNet: Rethinking Model Scaling for CNNs"**, ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Liu, Z. et al., **"A ConvNet for the 2020s"** (ConvNeXt), CVPR 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Huang, G. et al., **"Densely Connected Convolutional Networks"** (DenseNet), CVPR 2017. arXiv:1608.06993

Training-recipe references:
- Loshchilov, I. & Hutter, F., **"Decoupled Weight Decay Regularization"** (AdamW), ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Zhang, H. et al., **"mixup: Beyond Empirical Risk Minimization"**, ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- Yun, S. et al., **"CutMix"**, ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- Cubuk, E. D. et al., **"RandAugment"**, NeurIPS 2020. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
- Müller, S. & Hutter, F., **"TrivialAugment"**, ICCV 2021. [arXiv:2103.10158](https://arxiv.org/abs/2103.10158)
- Lin, T.-Y. et al., **"Focal Loss for Dense Object Detection"**, ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Szegedy, C. et al., **"Rethinking the Inception Architecture"** (label smoothing), CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Izmailov, P. et al., **"Averaging Weights Leads to Wider Optima and Better Generalization"** (SWA), UAI 2018. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)

Medical-imaging transfer learning:
- Raghu, M., Zhang, C., Kleinberg, J. & Bengio, S., **"Transfusion: Understanding Transfer Learning for Medical Imaging"**, NeurIPS 2019. [arXiv:1902.07208](https://arxiv.org/abs/1902.07208)
- Mei, X. et al., **"RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"**, *Radiology: AI* 2022. [doi.org/10.1148/ryai.210315](https://pubs.rsna.org/doi/full/10.1148/ryai.210315)

Data-leakage / split hazards (critical):
- Yagis, E. et al., **"Effect of data leakage in brain MRI classification using 2D convolutional neural networks"**, *Sci Rep* 11:22544, 2021. [doi.org/10.1038/s41598-021-01681-w](https://www.nature.com/articles/s41598-021-01681-w)
- Wen, J. et al., **"Convolutional Neural Networks for Classification of Alzheimer's Disease: Overview and Reproducible Evaluation"**, *MedIA* 2020. [arXiv:1904.07773](https://arxiv.org/abs/1904.07773)
- Hernández et al., **"Data Leakage in Deep Learning for Alzheimer's Disease Diagnosis: A Scoping Review of Methodological Rigor and Performance Inflation"**, *Diagnostics* 15(18):2348, 2025. [doi.org/10.3390/diagnostics15182348](https://www.mdpi.com/2075-4418/15/18/2348)
- Tampu, I. E. et al., **"Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images"**, *Sci Data* 2022. [arXiv:2202.12267](https://arxiv.org/abs/2202.12267)

Datasets:
- Kumar, S. & Shastri, S., **"Alzheimer MRI Preprocessed Dataset"**, Kaggle, 2022. [DOI 10.34740/kaggle/ds/2029496](https://commons.datacite.org/doi.org/10.34740/kaggle/ds/2029496) — original 6,400 source set.
- Borhanitrash, **"Alzheimer MRI Disease Classification Dataset"**, Kaggle. — the 44 K augmented variant we are using.

---

## 6. What success looks like

A clean comparison table on a leakage-controlled test set, where the
**best efficientnet_b0 config** beats the **clean baseline_cnn** entry by
at least 5 points of macro F1, on the same data, with all artifacts saved
and the code reviewer's notes addressed. The three-config ablation
(D-a / D-b / D-c) lets us attribute the gain to architecture+resolution,
Mixup, and SWA+TTA respectively. If we don't beat baseline_cnn, we report
the honest negative result and explain why.

## 7. Reviewer audit (Reviewer pass on this plan)

The reviewer agent found three MUST-FIX items, all addressed above:

1. **Mixup + label smoothing collision** → Phase C3 makes them mutually
   exclusive; default config uses Mixup, smoothing off.
2. **pHash variant under-specified** → Phase A1 pins
   `imagehash==4.3.1` + `imagehash.phash` 64-bit; A2 calibrates threshold
   against known md5 duplicates.
3. **Stratified split on `cluster_id` doesn't guarantee class balance**
   → Phase A4 stratifies per-class over clusters, weighted by image count;
   A5 adds explicit class-balance sanity check.

SHOULD-CONSIDER items applied:

4. Single full-recipe run replaced with three-config ablation
   (D-a / D-b / D-c) for attributable gains.
5. Time budget made explicit (~3–4.5 h total).
6. Phase B inference forced to 128 × 128 to avoid attributing
   resolution-induced drops to leakage.

Deferred:

7. RadImageNet pretraining as parallel run — saved for after the
   primary EfficientNet-B0 result lands; we do not block on it.
