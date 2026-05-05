"""
Phase A — perceptual-hash deduplication and clean-split construction.

Pipeline (idempotent; each step is its own callable so the notebook can
inspect intermediate state):

    A1: hash_all_images(...)         -> outputs/splits/image_hashes.parquet
    A2: build_clusters(hashes, k)    -> dict + cluster id added to df
    A3: leakage_audit(old, clusters) -> dict
    A4: build_clean_splits(clusters) -> writes train_clean.csv etc.
    A5: sanity_checks(...)           -> dict

The pHash variant is pinned to ``imagehash.phash`` at 64-bit (8x8 grid).
``imagehash==4.3.1`` and ``Pillow>=10`` are required by the calling notebook.
"""
from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, DATA_ROOT, REPORTS_DIR, SEED, SPLITS_DIR
from src.io_utils import get_logger, list_images


PHASH_BITS = 256       # 16x16 grid = 256 bits — gives much better separation
                       # at scale than 64-bit (which collapses MRI slices into
                       # mega-clusters via transitive merges).
PHASH_GRID = 16        # imagehash.phash uses an 8N x 8N DCT grid; size=16 -> 256 bits
DEFAULT_HAMMING = 16   # tunable; calibrated against known md5 duplicates in A2.
                       # For 256-bit phash, ~6% of bits.


# ----------------------------------------------------------------------------
# A1 — hash every image
# ----------------------------------------------------------------------------
def hash_all_images(out_path: Path | str | None = None,
                    sample_n: int | None = None) -> pd.DataFrame:
    """Compute imagehash.phash for every image in the dataset.

    Parameters
    ----------
    out_path : optional path to a .parquet file. Defaults to outputs/splits/image_hashes.parquet.
    sample_n : if set, hash only the first ``sample_n`` files per class (for
               quick smoke-tests; production run leaves it None).
    """
    log = get_logger("dedup.hash")
    df = list_images(DATA_ROOT)
    if sample_n is not None:
        # pandas 3 groupby+apply drops the grouping column from the result; just
        # take the first ``sample_n`` per class via boolean indexing instead.
        keep = df.groupby("class", sort=False).cumcount() < sample_n
        df = df[keep].copy()
    df = df.reset_index(drop=True)
    # Defensive: re-attach `class` if anything dropped it.
    if "class" not in df.columns and "label" in df.columns:
        df["class"] = df["label"].map({i: n for i, n in enumerate(CLASS_NAMES)})
    log.info("Hashing %d images...", len(df))
    t0 = time.time()
    phashes: list[str] = []
    md5s: list[str] = []
    sizes: list[str] = []
    for i, p in enumerate(df["filepath"].tolist()):
        try:
            with Image.open(p) as im:
                im.load()
                size = im.size
                # 256-bit phash: hash_size=16 means a 16x16 DCT-low-freq grid,
                # giving 256 bits of precision and much better discrimination
                # on visually-similar brain MRI than the 64-bit default.
                ph = imagehash.phash(im, hash_size=PHASH_GRID)
            data = Path(p).read_bytes()
            md5 = hashlib.md5(data).hexdigest()
        except Exception as e:
            log.warning("hash failed for %s: %s", p, e)
            phashes.append("")
            md5s.append("")
            sizes.append("0x0")
            continue
        phashes.append(str(ph))
        md5s.append(md5)
        sizes.append(f"{size[0]}x{size[1]}")
        if (i + 1) % 5000 == 0:
            log.info("  %d / %d  (%.1fs)", i + 1, len(df), time.time() - t0)
    df = df.copy()
    df["phash"] = phashes
    df["md5"] = md5s
    df["size"] = sizes
    log.info("Hashed %d images in %.1fs", len(df), time.time() - t0)
    out = Path(out_path or (SPLITS_DIR / "image_hashes.csv"))
    # Use CSV (no pyarrow dep) — file is small, ~2-3 MB for 44k images
    df.to_csv(out, index=False)
    log.info("Saved hashes to %s", out)
    return df


# ----------------------------------------------------------------------------
# A2 — cluster near-duplicates with union-find
# ----------------------------------------------------------------------------
def _phash_to_uint64s(hex_str: str, n_words: int) -> np.ndarray:
    """Decode a hex pHash into ``n_words`` uint64s (little-endian).

    256-bit pHash -> 4 uint64 words. Returns zeros for empty input."""
    if not hex_str:
        return np.zeros(n_words, dtype=np.uint64)
    # imagehash gives MSB-first hex; we keep that and split into n_words chunks
    pad = n_words * 16 - len(hex_str)
    if pad > 0:
        hex_str = "0" * pad + hex_str
    out = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        chunk = hex_str[w * 16:(w + 1) * 16]
        out[w] = np.uint64(int(chunk, 16))
    return out


def build_clusters(df: pd.DataFrame, hamming_thresh: int = DEFAULT_HAMMING,
                   bits: int = PHASH_BITS) -> pd.DataFrame:
    """Cluster images by Hamming distance over their pHash.

    For ``bits`` total, we represent each phash as ``W = bits // 64`` uint64
    words. We bucket by 16-bit bands across all words (so a 256-bit hash gets
    16 bands). If two phashes agree on at least one band, they're a candidate
    pair — provably true if their Hamming distance is below
    ``(num_bands) * 16 - 16`` (in our case 256-16 = 240, much larger than any
    realistic threshold). Then we do exact popcount on the candidate pairs.

    Naive O(N^2) over 44 k images is way too slow; banded bucketing keeps
    candidate pairs to a few million.
    """
    log = get_logger("dedup.cluster")
    df = df.copy().reset_index(drop=True)
    n = len(df)
    words = bits // 64

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    log.info("Converting %d phashes to %d-word uint64...", n, words)
    h_ints = np.zeros((n, words), dtype=np.uint64)
    for i, s in enumerate(df["phash"].tolist()):
        h_ints[i] = _phash_to_uint64s(s, words)

    # Bucket by 16-bit bands across all words (256-bit -> 16 bands).
    log.info("Bucketing by %d-band LSH...", words * 4)
    buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for i in range(n):
        h = h_ints[i]
        for w in range(words):
            for sub in range(4):
                key = (w, sub, int((h[w] >> np.uint64(sub * 16)) & np.uint64(0xFFFF)))
                buckets[key].append(i)

    def _popcount_u64(x: np.ndarray) -> np.ndarray:
        """Hamming-weight (popcount) for a uint64 array. Returns int64 with the
        same shape (last axis preserved so caller can sum across words)."""
        x = x.astype(np.uint64, copy=True)
        m1 = np.uint64(0x5555555555555555)
        m2 = np.uint64(0x3333333333333333)
        m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
        h01 = np.uint64(0x0101010101010101)
        x = x - ((x >> np.uint64(1)) & m1)
        x = (x & m2) + ((x >> np.uint64(2)) & m2)
        x = (x + (x >> np.uint64(4))) & m4
        return ((x * h01) >> np.uint64(56)).astype(np.int64)

    log.info("Pairwise compare within %d buckets...", len(buckets))
    pair_count = 0
    union_count = 0
    seen_pairs: set[tuple[int, int]] = set()  # dedupe pairs across multiple banded buckets
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        if len(members) > 8000:
            members = members[:8000]
        m = np.asarray(members, dtype=np.int64)
        h_sub = h_ints[m]   # (B, words)
        for i_idx in range(len(m)):
            anchor = h_sub[i_idx]                      # (words,)
            others = h_sub[i_idx + 1:]                 # (B-1, words)
            xor = np.bitwise_xor(others, anchor)        # (B-1, words)
            dist = _popcount_u64(xor).sum(axis=1)       # (B-1,)
            close = np.where(dist <= hamming_thresh)[0]
            for j_local in close:
                a = int(m[i_idx]); b = int(m[i_idx + 1 + j_local])
                pair_key = (a, b) if a < b else (b, a)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pair_count += 1
                if find(a) != find(b):
                    union(a, b)
                    union_count += 1

    # Compress paths and assign cluster IDs
    roots = [find(i) for i in range(n)]
    root_to_cid: dict[int, int] = {}
    cluster_ids = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(roots):
        if r not in root_to_cid:
            root_to_cid[r] = len(root_to_cid)
        cluster_ids[i] = root_to_cid[r]
    df["cluster_id"] = cluster_ids
    log.info("Built %d clusters from %d images (%d close-pairs, %d unions)",
             df["cluster_id"].nunique(), n, pair_count, union_count)
    return df


def cluster_size_stats(df_with_clusters: pd.DataFrame) -> dict:
    sizes = df_with_clusters.groupby("cluster_id").size().values
    by_class_max = (df_with_clusters.groupby(["class", "cluster_id"])
                                  .size().groupby(level=0).max().to_dict())
    return {
        "num_clusters": int(df_with_clusters["cluster_id"].nunique()),
        "max_cluster_size": int(sizes.max()),
        "mean_cluster_size": float(sizes.mean()),
        "median_cluster_size": float(np.median(sizes)),
        "p99_cluster_size": float(np.percentile(sizes, 99)),
        "max_per_class": {k: int(v) for k, v in by_class_max.items()},
    }


# ----------------------------------------------------------------------------
# A2 helper — calibrate against md5 duplicates
# ----------------------------------------------------------------------------
def calibrate_threshold_against_md5(hashes_df: pd.DataFrame,
                                    candidate_thresholds=(2, 4, 6, 8, 10, 12)) -> dict:
    """For each candidate threshold, build clusters and report:

    * how many md5-duplicate pairs land in the same cluster (recall on a
      ground-truth set the publisher may have created when copy-pasting);
    * cluster-size distribution stats.

    Use this to pick a threshold that recalls ~all md5 duplicates without
    creating mega-clusters.
    """
    log = get_logger("dedup.calibrate")

    # Find md5 duplicate pairs
    md5_groups = hashes_df.groupby("md5").size()
    md5_dup_groups = md5_groups[md5_groups > 1].index.tolist()
    md5_pairs = []
    for m in md5_dup_groups:
        idxs = hashes_df.index[hashes_df["md5"] == m].tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                md5_pairs.append((idxs[i], idxs[j]))
    log.info("Found %d md5-identical pairs in %d groups",
             len(md5_pairs), len(md5_dup_groups))

    out = {"md5_pairs": len(md5_pairs), "md5_groups": len(md5_dup_groups), "thresholds": {}}
    for t in candidate_thresholds:
        clustered = build_clusters(hashes_df, hamming_thresh=t)
        # recall on md5 ground truth
        recall_hits = 0
        for a, b in md5_pairs:
            if clustered.iloc[a]["cluster_id"] == clustered.iloc[b]["cluster_id"]:
                recall_hits += 1
        recall = recall_hits / max(len(md5_pairs), 1)
        stats = cluster_size_stats(clustered)
        out["thresholds"][t] = {
            "md5_recall": round(recall, 4),
            **stats,
        }
        log.info("  Hamming<=%d: recall=%.3f num_clusters=%d max_cluster=%d",
                 t, recall, stats["num_clusters"], stats["max_cluster_size"])
    return out


# ----------------------------------------------------------------------------
# A3 — leakage audit on the existing splits
# ----------------------------------------------------------------------------
def leakage_audit(df_with_clusters: pd.DataFrame,
                  splits=("train", "val", "test")) -> dict:
    """For each pair of existing splits, count cluster_ids that appear in both."""
    relpath_to_cid = dict(zip(df_with_clusters["relpath"], df_with_clusters["cluster_id"]))
    by_split = {}
    for s in splits:
        sdf = pd.read_csv(SPLITS_DIR / f"{s}.csv")
        cids = sorted({relpath_to_cid[r] for r in sdf["relpath"].tolist()
                       if r in relpath_to_cid})
        by_split[s] = set(cids)
        by_split[f"{s}_n_clusters"] = len(cids)
        by_split[f"{s}_n_images"] = len(sdf)

    overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            inter = by_split[a] & by_split[b]
            overlaps[f"{a}<->{b}"] = {
                "shared_clusters": len(inter),
                "share_of_{a}".replace("{a}", a): round(len(inter) / max(len(by_split[a]), 1), 4),
                "share_of_{b}".replace("{b}", b): round(len(inter) / max(len(by_split[b]), 1), 4),
            }

    return {
        "per_split": {s: {"n_images": by_split[f"{s}_n_images"],
                          "n_clusters": by_split[f"{s}_n_clusters"]}
                      for s in splits},
        "overlaps": overlaps,
    }


# ----------------------------------------------------------------------------
# A4 — build clean cluster-stratified splits
# ----------------------------------------------------------------------------
def build_clean_splits(df_with_clusters: pd.DataFrame,
                        train_frac: float = 0.70, val_frac: float = 0.15,
                        test_frac: float = 0.15, seed: int = SEED) -> dict:
    """Cluster-level stratified split.

    For each *class*, take all clusters whose dominant label is that class
    and assign them to {train, val, test} in proportions train/val/test by
    *image count* (not cluster count). Greedy fill: shuffle clusters with
    a fixed RNG and place each cluster into the split whose current
    cumulative share is most below target. This handles size-skewed
    clusters without putting one mega-cluster in test.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.")
    log = get_logger("dedup.split")
    rng = np.random.default_rng(seed)

    # Each cluster's dominant class
    cluster_class = (df_with_clusters.groupby("cluster_id")["label"]
                     .agg(lambda s: int(s.mode().iloc[0])))
    cluster_size = df_with_clusters.groupby("cluster_id").size()

    df_with_clusters = df_with_clusters.copy()
    df_with_clusters["split"] = ""

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_clusters = cluster_class[cluster_class == cls_idx].index.tolist()
        rng.shuffle(cls_clusters)  # deterministic
        sizes = cluster_size.loc[cls_clusters].values.astype(int)
        total = sizes.sum()
        target = {
            "train": train_frac * total,
            "val": val_frac * total,
            "test": test_frac * total,
        }
        accum = {"train": 0, "val": 0, "test": 0}
        # Place biggest clusters first to spread them across splits sanely
        order = np.argsort(sizes)[::-1]
        cluster_to_split: dict[int, str] = {}
        for k in order:
            cid = cls_clusters[int(k)]
            sz = int(sizes[k])
            # pick split with biggest remaining capacity (target - accum) ratio
            choice = max(target.keys(), key=lambda s: (target[s] - accum[s]))
            cluster_to_split[cid] = choice
            accum[choice] += sz

        # Assign rows
        mask = df_with_clusters["cluster_id"].isin(cls_clusters)
        df_with_clusters.loc[mask, "split"] = (
            df_with_clusters.loc[mask, "cluster_id"].map(cluster_to_split)
        )
        log.info("Class %s: %d clusters %d images -> "
                 "train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%)",
                 cls_name, len(cls_clusters), total,
                 accum["train"], 100 * accum["train"] / total,
                 accum["val"], 100 * accum["val"] / total,
                 accum["test"], 100 * accum["test"] / total)

    out = {}
    for s in ("train", "val", "test"):
        sdf = df_with_clusters[df_with_clusters["split"] == s].copy()
        path = SPLITS_DIR / f"{s}_clean.csv"
        sdf[["relpath", "class", "label", "cluster_id"]].to_csv(path, index=False)
        out[s] = {"path": str(path), "n_images": len(sdf),
                  "n_clusters": int(sdf["cluster_id"].nunique())}
    return out


# ----------------------------------------------------------------------------
# A5 — sanity checks
# ----------------------------------------------------------------------------
def sanity_checks(df_with_clusters: pd.DataFrame, expected_class_share=None) -> dict:
    """1) No cluster crosses splits. 2) Test-set per-class within ±10 pp."""
    out = {"errors": []}
    if expected_class_share is None:
        # Compute from full dataset
        full_counts = df_with_clusters["class"].value_counts(normalize=True).to_dict()
        expected_class_share = full_counts

    # 1: no cluster in two splits
    cluster_split_pairs = (df_with_clusters[["cluster_id", "split"]]
                           .drop_duplicates())
    bad = (cluster_split_pairs.groupby("cluster_id").size())
    bad = bad[bad > 1]
    if len(bad) > 0:
        out["errors"].append(f"{len(bad)} cluster_ids span multiple splits")

    # 2: test set class share
    test_df = df_with_clusters[df_with_clusters["split"] == "test"]
    test_share = (test_df["class"].value_counts(normalize=True)
                                  .reindex(expected_class_share.keys()).fillna(0).to_dict())
    deltas = {k: round(test_share[k] - expected_class_share[k], 4)
              for k in expected_class_share}
    out["test_class_share"] = test_share
    out["test_class_share_delta_vs_dataset"] = deltas
    for k, d in deltas.items():
        if abs(d) > 0.10:
            out["errors"].append(
                f"class {k} test share {test_share[k]:.3f} differs from "
                f"dataset {expected_class_share[k]:.3f} by {d:+.3f} (>0.10)"
            )

    # 3: every (split, class) pair has a non-degenerate count.
    # Flags the regression case where a tighter Hamming threshold collapses
    # one class into a few mega-clusters that all land in train.
    for s in ("train", "val", "test"):
        sdf = df_with_clusters[df_with_clusters["split"] == s]
        for cls in expected_class_share:
            n_in_split = int((sdf["class"] == cls).sum())
            full_n = int((df_with_clusters["class"] == cls).sum())
            target = {"train": 0.70, "val": 0.15, "test": 0.15}[s]
            min_acceptable = 0.50 * target * full_n
            if n_in_split < min_acceptable:
                out["errors"].append(
                    f"split={s} class={cls} has only {n_in_split} images "
                    f"(< {min_acceptable:.0f} = 50%% of target {target * full_n:.0f})"
                )

    # cluster size distribution per class
    out["cluster_size_stats"] = cluster_size_stats(df_with_clusters)
    return out


def run_phase_a(hamming_thresh: int = DEFAULT_HAMMING) -> dict:
    """End-to-end driver for the notebook to call."""
    log = get_logger("dedup.phaseA", logfile=REPORTS_DIR / "phaseA_dedup.log")
    log.info("=== Phase A — pHash dedup, threshold=%d ===", hamming_thresh)

    hashes_path = SPLITS_DIR / "image_hashes.csv"
    if hashes_path.exists():
        log.info("Reading cached hashes: %s", hashes_path)
        hashes_df = pd.read_csv(hashes_path)
    else:
        hashes_df = hash_all_images()

    clustered = build_clusters(hashes_df, hamming_thresh=hamming_thresh)
    leakage = leakage_audit(clustered)
    # build_clean_splits writes split CSVs and also returns the assignment;
    # we re-load to get the per-row split label for sanity_checks.
    splits_summary = build_clean_splits(clustered)
    relpath_to_split: dict[str, str] = {}
    for s in ("train", "val", "test"):
        sdf = pd.read_csv(SPLITS_DIR / f"{s}_clean.csv")
        for r in sdf["relpath"].tolist():
            relpath_to_split[r] = s
    clustered = clustered.copy()
    clustered["split"] = clustered["relpath"].map(relpath_to_split)
    sanity = sanity_checks(clustered)

    summary = {
        "hamming_threshold": hamming_thresh,
        "n_images": int(len(clustered)),
        "cluster_stats": cluster_size_stats(clustered),
        "leakage_audit_old_split": leakage,
        "clean_split_summary": splits_summary,
        "sanity": sanity,
    }
    out = REPORTS_DIR / "phaseA_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Phase A summary saved to %s", out)
    return summary


if __name__ == "__main__":
    s = run_phase_a()
    print(json.dumps(s, indent=2))
