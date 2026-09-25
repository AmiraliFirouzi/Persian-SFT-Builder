"""
Build a quality-aware, diversity-preserving 25K subset from:
    xmanii/Maux-Persian-SFT-30k

Selection recipe
----------------
1) Load 30K train split.
2) Parse/validate messages; audit invalid structure and missing score separately.
3) Exact dedup.
4) Near-duplicate detection with MinHash + LSH; audit candidate pairs and
   cross-source candidate pairs rather than treating "0 removed" as proof of
   zero redundancy.
5) Quality-aware source allocation (quality priority + diversity floor/cap), with explicit quality-evidence metadata for unscored sources.
   Source size is used for availability, not proportional quota.
6) Within each source, stratify by:
      score percentile bucket × heuristic task bucket
   Score buckets are source-local:
      P00-P05, P05-P25, P25-P50, P50-P75, P75-P95, P95-P100
   Missing-score rows live in a separate "MISSING_SCORE" bucket and are NEVER
   assigned a synthetic score or entered into quality ranking.
7) Allocate the source quota across buckets proportional to that source's
   bucket distribution.
8) Inside each bucket, sample WITHOUT replacement using quality-weighted random
   probabilities rather than head(n). This gives high-quality examples a higher
   chance while preserving bucket diversity.
9) Run an independent TF-IDF char-ngram nearest-neighbor sanity audit on a fixed sample.
10) Save final data plus detailed audit files.

Mauxi-SFT-Persian is NOT loaded separately because it is already represented as
one source inside Maux-Persian-SFT-30k.

Optional exact tokenizer audit
------------------------------
Set TOKENIZER_NAME to the tokenizer used by the target Qwen model. When None,
the script uses a deterministic whitespace/punctuation approximation and labels
it as approximate in the audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from datasketch import MinHash, MinHashLSH


# =========================
# Configuration
# =========================
DATASET_ID = "xmanii/Maux-Persian-SFT-30k"
SPLIT = "train"
TARGET_N = 25_000
SEED = 42

# Near-duplicate settings.
NUM_PERM = 128
SHINGLE_SIZE = 5
NEAR_DUP_THRESHOLDS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)
LSH_MIN_THRESHOLD = 0.85

# Source allocation.
MIN_SOURCE_FRACTION = 0.005   # 0.5% floor when source has enough rows
MAX_SOURCE_FRACTION = 0.32    # safety cap, not a proportional quota
SOURCE_PRIORITY_POWER = 0.75
QUALITY_TOP_FRACTION = 0.30
GLOBAL_SCORE_WEIGHT = 0.80
SOURCE_RANK_WEIGHT = 0.20
ROBUST_Q_LOW = 0.05
ROBUST_Q_HIGH = 0.95

# Stratified selection.
# Missing-score rows are handled as a separate bucket. There is deliberately
# no hidden cap by default: we measure their final share first instead of
# inventing a synthetic score. Set to e.g. 0.10 after reviewing the audit if
# the project policy requires a maximum.
MAX_MISSING_SCORE_FRACTION: float | None = None

# Quality-weighted random selection inside each bucket.
# Larger values make higher-quality rows more likely, but never deterministic.
QUALITY_SAMPLING_TEMPERATURE = 4.0

# Optional exact tokenizer for audit. Example:
# TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
TOKENIZER_NAME: str | None = None

# Heuristic task classification. This is an audit feature because the source
# dataset does not expose a canonical task column.
TASK_LABELS = (
    "function_calling",
    "coding",
    "math_reasoning",
    "reasoning",
    "translation",
    "summarization",
    "creative_writing",
    "classification_extraction",
    "instructional_qa",
    "other",
)

OUTPUT_DIR = Path("maux_persian_sft_25k_v7")
OUTPUT_PARQUET = OUTPUT_DIR / "maux_persian_sft_25k_v7.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "maux_persian_sft_25k_v7.jsonl"
OUTPUT_SOURCE_AUDIT = OUTPUT_DIR / "maux_persian_sft_25k_v7_source_audit.csv"
OUTPUT_SCORE_AUDIT = OUTPUT_DIR / "maux_persian_sft_25k_v7_score_distribution.csv"
OUTPUT_TASK_AUDIT = OUTPUT_DIR / "maux_persian_sft_25k_v7_task_distribution.csv"
OUTPUT_TOKEN_AUDIT = OUTPUT_DIR / "maux_persian_sft_25k_v7_token_audit.csv"
OUTPUT_AUDIT_SUMMARY = OUTPUT_DIR / "maux_persian_sft_25k_v7_audit_summary.json"
OUTPUT_TASK_MULTILABEL_AUDIT = OUTPUT_DIR / "maux_persian_sft_25k_v7_task_multilabel_audit.csv"

# Independent near-duplicate sanity audit. This does NOT replace MinHash/LSH.
SANITY_AUDIT_SAMPLE_N = 5_000
SANITY_TFIDF_NGRAM_RANGE = (3, 5)
SANITY_TFIDF_MAX_FEATURES = 200_000
SANITY_NEIGHBORS = 50


# =========================
# Persian/text helpers
# =========================
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
_WHITESPACE_RE = re.compile(r"\s+")
_PERSIAN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).translate(_ARABIC_TO_PERSIAN)
    text = text.replace("\u200e", "").replace("\u200f", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


SEMANTIC_MESSAGE_KEYS = (
    "role",
    "content",
    "name",
    "tool_calls",
    "function_call",
)


def normalize_for_dedup(value: object) -> object:
    """Recursively normalize all information-bearing dedup fields.

    Dict keys are sorted for deterministic hashing. List/tuple ordering is
    intentionally preserved because tool-call order can be semantically
    meaningful.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            str(k): normalize_for_dedup(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_for_dedup(v) for v in value]
    if isinstance(value, str):
        return normalize_text(value)
    return value


def canonicalize_messages(messages: object) -> list[dict[str, object]]:
    if messages is None:
        return []
    if not isinstance(messages, (list, tuple)):
        try:
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
        except Exception:
            pass
    if not isinstance(messages, (list, tuple)):
        return []

    result: list[dict[str, object]] = []
    for message in messages:
        if message is None:
            continue
        if not isinstance(message, dict):
            try:
                if hasattr(message, "as_py"):
                    message = message.as_py()
            except Exception:
                pass
        if not isinstance(message, dict):
            continue

        normalized = normalize_for_dedup(message)
        if not isinstance(normalized, dict):
            continue

        role = normalize_text(normalized.get("role", "")).lower()
        cleaned = {
            key: normalized[key]
            for key in SEMANTIC_MESSAGE_KEYS
            if key in normalized
        }
        cleaned["role"] = role
        if "content" in cleaned and isinstance(cleaned["content"], str):
            cleaned["content"] = normalize_text(cleaned["content"])
        result.append(cleaned)
    return result


def has_trainable_assistant(messages: list[dict[str, object]]) -> bool:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = normalize_for_dedup(message.get("content", ""))
        if isinstance(content, str):
            content_ok = bool(content.strip())
        else:
            content_ok = bool(content)
        tool_calls_ok = bool(message.get("tool_calls"))
        function_call_ok = bool(message.get("function_call"))
        if content_ok or tool_calls_ok or function_call_ok:
            return True
    return False


def has_sft_structure(messages: list[dict[str, object]]) -> bool:
    roles = {str(m.get("role", "")) for m in messages}
    return bool(messages) and "user" in roles and has_trainable_assistant(messages)


def canonical_key(messages: list[dict[str, object]]) -> str:
    payload = json.dumps(
        normalize_for_dedup(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def message_to_audit_text(message: dict[str, object]) -> str:
    payload = {
        key: message[key]
        for key in SEMANTIC_MESSAGE_KEYS
        if key in message
    }
    return json.dumps(
        normalize_for_dedup(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def conversation_text(messages: list[dict[str, object]]) -> str:
    return "\n".join(message_to_audit_text(m) for m in messages)


def word_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = re.findall(r"\S+", text)
    if len(words) <= k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM, seed=SEED)
    shingles = word_shingles(text)
    if not shingles:
        shingles = {"__EMPTY__"}
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def inspect_message_schema(df: pd.DataFrame) -> None:
    sample = df["messages"].head(25)
    type_counts = sample.map(lambda x: type(x).__name__).value_counts().to_dict()
    usable = 0
    role_counts: dict[str, int] = {}
    for value in sample:
        msgs = canonicalize_messages(value)
        if msgs:
            usable += 1
        for msg in msgs:
            role = msg.get("role", "")
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
    print("  message value types (first 25):", type_counts)
    print("  non-empty canonical messages (first 25):", usable)
    print("  roles observed (first 25):", dict(sorted(role_counts.items())))
    if usable == 0:
        raise RuntimeError("Could not parse the 'messages' column.")


# =========================
# Exact / near dedup
# =========================
def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_canonical_messages"] = work["messages"].map(canonicalize_messages)
    work = work[work["_canonical_messages"].map(has_sft_structure)].copy()
    work["_dedup_key"] = work["_canonical_messages"].map(canonical_key)
    work = work.sort_values(
        by=["score", "original_index"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    )
    return work.drop_duplicates(subset="_dedup_key", keep="first").copy()


def near_dedup(df: pd.DataFrame, near_dup_threshold: float) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df.copy().sort_values(
        by=["score", "original_index"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    )

    lsh_threshold = max(LSH_MIN_THRESHOLD, near_dup_threshold - 0.05)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=NUM_PERM)
    kept_minhashes: dict[str, MinHash] = {}
    kept_sources: dict[str, str] = {}
    kept_indices: list[Any] = []

    candidate_matches = 0
    candidate_cross_source = 0
    confirmed_duplicates = 0
    confirmed_cross_source = 0

    for row_number, (idx, row) in enumerate(work.iterrows()):
        mh = build_minhash(conversation_text(row["_canonical_messages"]))
        candidates = lsh.query(mh)
        row_source = str(row["source"])
        if candidates:
            candidate_matches += len(candidates)
            candidate_cross_source += sum(
                1 for key in candidates if kept_sources.get(key) != row_source
            )

        duplicate_keys = [
            key for key in candidates
            if mh.jaccard(kept_minhashes[key]) >= near_dup_threshold
        ]
        if duplicate_keys:
            confirmed_duplicates += 1
            if any(kept_sources.get(key) != row_source for key in duplicate_keys):
                confirmed_cross_source += 1
            continue

        key = f"doc_{row_number}"
        lsh.insert(key, mh)
        kept_minhashes[key] = mh
        kept_sources[key] = row_source
        kept_indices.append(idx)

    stats = {
        "candidate_matches": int(candidate_matches),
        "candidate_cross_source_matches": int(candidate_cross_source),
        "confirmed_near_duplicate_rows_removed": int(confirmed_duplicates),
        "confirmed_cross_source_near_duplicate_rows_removed": int(confirmed_cross_source),
    }
    return work.loc[kept_indices].copy(), stats


# =========================
# Quality representation
# =========================
def compute_quality_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()
    score = pd.to_numeric(work["score"], errors="coerce")
    valid = score.notna()
    if not valid.any():
        raise RuntimeError("No numeric scores are available in the deduplicated dataset.")

    low = float(score[valid].quantile(ROBUST_Q_LOW))
    high = float(score[valid].quantile(ROBUST_Q_HIGH))
    if high <= low:
        low = float(score[valid].min())
        high = float(score[valid].max())

    if high <= low:
        global_quality = pd.Series(0.5, index=work.index, dtype=float)
    else:
        clipped = score.clip(lower=low, upper=high)
        global_quality = ((clipped - low) / (high - low)).astype(float)

    # Within-source percentile uses ONLY scored rows. Missing scores remain NaN.
    source_rank = pd.Series(np.nan, index=work.index, dtype=float)
    scored_work = work.loc[valid, ["source", "score"]].copy()
    source_rank.loc[valid] = scored_work.groupby("source", sort=False)["score"].rank(method="average", pct=True)

    work["_global_quality"] = global_quality
    work["_source_rank_quality"] = source_rank
    work["_quality"] = (
        GLOBAL_SCORE_WEIGHT * global_quality
        + SOURCE_RANK_WEIGHT * source_rank
    ).clip(0.0, 1.0)
    work.loc[~valid, ["_global_quality", "_source_rank_quality", "_quality"]] = np.nan
    return work, {"robust_low": low, "robust_high": high}


def build_source_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(df["_quality"].mean()) if df["_quality"].notna().any() else 0.5
    rows = []
    for source, group in df.groupby("source", sort=True):
        q = group["_quality"].dropna().astype(float)
        missing_n = int(pd.to_numeric(group["score"], errors="coerce").isna().sum())
        if q.empty:
            mean_q = median_q = top_mean = None
            source_priority = global_mean
            quality_evidence = "none_prior_global_mean"
        else:
            top_n = max(1, int(np.ceil(len(q) * QUALITY_TOP_FRACTION)))
            mean_q = float(q.mean())
            median_q = float(q.median())
            top_mean = float(q.nlargest(top_n).mean())
            source_priority = 0.70 * top_mean + 0.30 * mean_q
            quality_evidence = "observed_score"
        rows.append({
            "source": source,
            "available_n": int(len(group)),
            "scored_n": int(q.size),
            "missing_score_n": missing_n,
            "missing_score_pct": round(missing_n / max(len(group), 1) * 100, 3),
            "mean_quality_scored": None if mean_q is None else round(mean_q, 6),
            "median_quality_scored": None if median_q is None else round(median_q, 6),
            "top30_mean_quality_scored": None if top_mean is None else round(top_mean, 6),
            "source_priority": round(float(source_priority), 6),
            "quality_evidence": quality_evidence,
        })
    return pd.DataFrame(rows).sort_values(
        ["source_priority", "source"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def quality_aware_source_allocation(source_quality: pd.DataFrame, target_n: int) -> pd.DataFrame:
    work = source_quality.copy()
    min_n = max(1, int(np.floor(target_n * MIN_SOURCE_FRACTION)))
    max_n = max(min_n, int(np.floor(target_n * MAX_SOURCE_FRACTION)))

    work["min_quota"] = np.minimum(work["available_n"], min_n).astype(int)
    work["max_quota"] = np.minimum(work["available_n"], max_n).astype(int)
    work["quota"] = work["min_quota"].copy()

    if int(work["min_quota"].sum()) > target_n:
        raise RuntimeError("Source minimum floors exceed the target.")
    if int(work["max_quota"].sum()) < target_n:
        raise RuntimeError("Source caps cannot accommodate the target.")

    remaining = int(target_n - work["quota"].sum())
    rows = work.reset_index(drop=True)
    while remaining > 0:
        eligible = rows[rows["quota"] < rows["max_quota"]].copy()
        priority = eligible["source_priority"].astype(float).clip(lower=1e-12)
        denom = eligible["quota"].astype(float) + 1.0
        eligible["allocation_priority"] = priority.pow(SOURCE_PRIORITY_POWER) / denom
        chosen = int(eligible["allocation_priority"].idxmax())
        rows.loc[chosen, "quota"] += 1
        remaining -= 1

    if int(rows["quota"].sum()) != target_n:
        raise RuntimeError("Source quota allocation did not sum to target.")
    return rows.sort_values("source", kind="stable").reset_index(drop=True)


# =========================
# Task classification and stratification
# =========================
_CODE_RE = re.compile(r"```|\b(def|class|import|from|return|async|await|javascript|typescript|python|sql|bash|java|c\+\+|rust)\b", re.I)
_MATH_RE = re.compile(r"(∑|√|π|≤|≥|=\s*\d|\b(?:معادله|ریاضی|حساب|احتمال|انتگرال|مشتق|حل کن)\b|\b(?:equation|integral|derivative|probability|math)\b)", re.I)
_TRANSLATION_RE = re.compile(r"\b(?:translate|translation)\b|(?:ترجمه|ترجمه کن|به انگلیسی|به فارسی)", re.I)
_SUMMARY_RE = re.compile(r"\b(?:summarize|summary)\b|(?:خلاصه|جمع‌بندی|خلاصه کن)", re.I)
_CREATIVE_RE = re.compile(r"(?:شعر|داستان|قصه|سناریو|متن خلاق|رمان|تبریک|creative writing|poem|story)", re.I)
_CLASSIFY_RE = re.compile(r"(?:طبقه[‌ ]?بندی|دسته[‌ ]?بندی|استخراج کن|اطلاعات را استخراج|classification|classify|extract)", re.I)
_REASON_RE = re.compile(r"(?:استدلال|مرحله به مرحله|گام به گام|چرا|تحلیل کن|reasoning|step by step|chain of thought)", re.I)
_FUNCTION_RE = re.compile(r"(?:tool_calls|function_call|function calling|arguments|\"function\"|\"tool\"|ابزار|فراخوانی تابع)", re.I)


def classify_task_multilabel(messages: list[dict[str, object]], source: str) -> dict[str, bool]:
    blob = conversation_text(messages)
    roles = {m.get("role", "") for m in messages}
    lowered_source = str(source).lower()
    return {
        "function_calling": bool(
            "tool" in roles
            or any(m.get("tool_calls") or m.get("function_call") for m in messages)
            or _FUNCTION_RE.search(blob)
            or "function-calling" in lowered_source
        ),
        "coding": bool(_CODE_RE.search(blob)),
        "math_reasoning": bool(_MATH_RE.search(blob)),
        "reasoning": bool("reasoning" in lowered_source or _REASON_RE.search(blob)),
        "translation": bool(_TRANSLATION_RE.search(blob)),
        "summarization": bool(_SUMMARY_RE.search(blob)),
        "creative_writing": bool(_CREATIVE_RE.search(blob)),
        "classification_extraction": bool(_CLASSIFY_RE.search(blob)),
        "instructional_qa": bool(
            "financial" in lowered_source
            or "instruction" in lowered_source
            or "qa" in lowered_source
        ),
    }


def classify_task(messages: list[dict[str, object]], source: str) -> str:
    labels = classify_task_multilabel(messages, source)
    precedence = [
        "function_calling",
        "coding",
        "math_reasoning",
        "reasoning",
        "translation",
        "summarization",
        "creative_writing",
        "classification_extraction",
        "instructional_qa",
    ]
    for label in precedence:
        if labels[label]:
            return label
    return "other"


def add_task_buckets(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    primary = []
    label_rows = []
    for msgs, source in zip(work["_canonical_messages"], work["source"]):
        labels = classify_task_multilabel(msgs, source)
        label_rows.append(labels)
        primary.append(classify_task(msgs, source))
    work["_task_bucket"] = primary
    for label in TASK_LABELS:
        if label == "other":
            continue
        work[f"_is_{label}"] = [int(row.get(label, False)) for row in label_rows]
    return work


def build_task_multilabel_audit(original: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in [("global", original), ("global_final", final)]:
        for label in TASK_LABELS:
            if label == "other":
                value = (frame["_task_bucket"] == "other").mean() if len(frame) else 0.0
            else:
                col = f"_is_{label}"
                value = frame[col].mean() if col in frame.columns and len(frame) else 0.0
            rows.append({
                "scope": scope,
                "source": "__ALL__",
                "task_label": label,
                "n": int(round(value * len(frame))) if len(frame) else 0,
                "pct": round(float(value) * 100, 3),
            })
    sources = sorted(set(original["source"]) | set(final["source"]))
    for source in sources:
        for scope, frame in [
            ("source_original", original[original["source"] == source]),
            ("source_final", final[final["source"] == source]),
        ]:
            for label in TASK_LABELS:
                if label == "other":
                    value = (frame["_task_bucket"] == "other").mean() if len(frame) else 0.0
                else:
                    col = f"_is_{label}"
                    value = frame[col].mean() if col in frame.columns and len(frame) else 0.0
                rows.append({
                    "scope": scope,
                    "source": source,
                    "task_label": label,
                    "n": int(round(value * len(frame))) if len(frame) else 0,
                    "pct": round(float(value) * 100, 3),
                })
    return pd.DataFrame(rows)


def source_score_bucket(score: pd.Series) -> pd.Series:
    """Source-local percentile buckets for scored rows."""
    result = pd.Series(index=score.index, dtype="object")
    valid = score.notna()
    if not valid.any():
        return result
    s = score.loc[valid]
    # rank(pct=True) handles ties robustly and avoids quantile collapse for flat scores.
    pct = s.rank(method="average", pct=True)
    labels = np.select(
        [pct <= 0.05, pct <= 0.25, pct <= 0.50, pct <= 0.75, pct <= 0.95],
        ["P00-P05", "P05-P25", "P25-P50", "P50-P75", "P75-P95"],
        default="P95-P100",
    )
    result.loc[valid] = labels
    result.loc[~valid] = "MISSING_SCORE"
    return result


def assign_stratification_buckets(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_score_bucket"] = (
        work.groupby("source", sort=False)["score"].transform(source_score_bucket)
    )
    work.loc[work["score"].isna(), "_score_bucket"] = "MISSING_SCORE"
    work["_strat_bucket"] = work["_score_bucket"].astype(str) + "__" + work["_task_bucket"].astype(str)
    return work


def largest_remainder_counts(counts: pd.Series, target_n: int) -> pd.Series:
    counts = counts.astype(int)
    total = int(counts.sum())
    if target_n <= 0:
        return pd.Series(0, index=counts.index, dtype=int)
    if target_n >= total:
        return counts.copy()

    raw = counts / total * target_n
    alloc = pd.Series(np.floor(raw.to_numpy()).astype(int), index=raw.index, dtype=int)
    remainder = int(target_n - alloc.sum())
    if remainder > 0:
        frac = (raw - alloc).sort_values(ascending=False, kind="stable")
        for key in frac.index[:remainder]:
            alloc.loc[key] += 1
    return alloc


def weighted_random_sample(group: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return group.iloc[0:0].copy()
    if n >= len(group):
        return group.copy()

    work = group.copy()
    q = work["_quality"].astype(float)
    # Missing score must never enter this function.
    if q.isna().any():
        raise RuntimeError("Missing-score rows entered quality-weighted sampling.")

    logits = QUALITY_SAMPLING_TEMPERATURE * (q.to_numpy() - float(q.mean()))
    logits = np.clip(logits, -30.0, 30.0)
    weights = np.exp(logits)
    weights = weights / weights.sum()
    chosen_positions = rng.choice(len(work), size=n, replace=False, p=weights)
    return work.iloc[np.sort(chosen_positions)].copy()


def stratified_source_sample(source_df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return source_df.iloc[0:0].copy()
    if n >= len(source_df):
        return source_df.copy()

    # First allocate quota across score × task buckets in proportion to their
    # source distribution.
    counts = source_df.groupby("_strat_bucket", sort=True).size()
    bucket_quota = largest_remainder_counts(counts, n)

    pieces: list[pd.DataFrame] = []
    for bucket, quota in bucket_quota.items():
        group = source_df[source_df["_strat_bucket"] == bucket]
        if len(group) == 0 or int(quota) == 0:
            continue
        if bucket.startswith("MISSING_SCORE"):
            # Missing score is not quality-ranked. Use deterministic random
            # sampling so it remains a genuine separate quality-unknown bucket.
            if int(quota) >= len(group):
                chosen = group.copy()
            else:
                pos = rng.choice(len(group), size=int(quota), replace=False)
                chosen = group.iloc[np.sort(pos)].copy()
        else:
            chosen = weighted_random_sample(group, int(quota), rng)
        pieces.append(chosen)

    final = pd.concat(pieces, ignore_index=False) if pieces else source_df.iloc[0:0].copy()
    # Largest remainder should already give n; safety fallback handles edge cases.
    if len(final) < n:
        selected_idx = set(final.index.tolist())
        remaining = source_df.loc[~source_df.index.isin(selected_idx)]
        if len(remaining) < n - len(final):
            raise RuntimeError("Stratified sampling fallback does not have enough remaining rows.")
        pos = rng.choice(len(remaining), size=n - len(final), replace=False)
        final = pd.concat([final, remaining.iloc[np.sort(pos)]], ignore_index=False)
    elif len(final) > n:
        pos = rng.choice(len(final), size=n, replace=False)
        final = final.iloc[np.sort(pos)].copy()
    return final.reset_index(drop=False).rename(columns={"index": "_selection_original_index"})


def independent_near_duplicate_sanity_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Independent TF-IDF char-ngram nearest-neighbor sanity check.

    This is intentionally separate from MinHash/LSH and is only a diagnostic.
    """
    sample_n = min(SANITY_AUDIT_SAMPLE_N, len(df))
    if sample_n < 3:
        return {"enabled": False, "reason": "fewer than 3 rows"}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        return {"enabled": False, "reason": f"scikit-learn unavailable: {exc}"}

    sample = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    texts = [conversation_text(x) for x in sample["_canonical_messages"]]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=SANITY_TFIDF_NGRAM_RANGE,
        min_df=2,
        max_features=SANITY_TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        return {"enabled": False, "reason": f"TF-IDF fit failed: {exc}"}
    k = min(SANITY_NEIGHBORS + 1, sample_n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    similarities = 1.0 - distances

    global_best = []
    same_source_best = []
    cross_source_best = []
    threshold_hits = {"0.85": 0, "0.90": 0, "0.95": 0}
    pair_hits: dict[tuple[int, int], float] = {}

    for i in range(sample_n):
        global_sims = []
        same_sims = []
        cross_sims = []
        for pos in range(1, k):  # skip self at position 0
            j = int(indices[i, pos])
            sim = float(similarities[i, pos])
            if j == i:
                continue
            global_sims.append(sim)
            if str(sample.loc[i, "source"]) == str(sample.loc[j, "source"]):
                same_sims.append(sim)
            else:
                cross_sims.append(sim)
            if sim >= 0.85:
                key = (min(i, j), max(i, j))
                pair_hits[key] = max(sim, pair_hits.get(key, 0.0))
        if global_sims:
            best = max(global_sims)
            global_best.append(best)
            if best >= 0.85:
                threshold_hits["0.85"] += 1
            if best >= 0.90:
                threshold_hits["0.90"] += 1
            if best >= 0.95:
                threshold_hits["0.95"] += 1
        if same_sims:
            same_source_best.append(max(same_sims))
        if cross_sims:
            cross_source_best.append(max(cross_sims))

    def summarize(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p95": None, "p99": None, "max": None}
        arr = np.asarray(values, dtype=float)
        return {
            "p95": round(float(np.quantile(arr, 0.95)), 6),
            "p99": round(float(np.quantile(arr, 0.99)), 6),
            "max": round(float(arr.max()), 6),
        }

    return {
        "enabled": True,
        "sample_n": sample_n,
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": list(SANITY_TFIDF_NGRAM_RANGE),
            "max_features": SANITY_TFIDF_MAX_FEATURES,
        },
        "neighbors": SANITY_NEIGHBORS,
        "global_best_similarity": summarize(global_best),
        "same_source_best_similarity": summarize(same_source_best),
        "cross_source_best_similarity": summarize(cross_source_best),
        "rows_with_best_neighbor_ge_0.85": int(threshold_hits["0.85"]),
        "rows_with_best_neighbor_ge_0.90": int(threshold_hits["0.90"]),
        "rows_with_best_neighbor_ge_0.95": int(threshold_hits["0.95"]),
        "unique_sample_pairs_ge_0.85": int(len(pair_hits)),
        "top_pair_examples": [
            {
                "similarity": round(float(sim), 6),
                "source_a": str(sample.loc[i, "source"]),
                "source_b": str(sample.loc[j, "source"]),
                "original_index_a": json_safe(sample.loc[i, "original_index"]),
                "original_index_b": json_safe(sample.loc[j, "original_index"]),
            }
            for (i, j), sim in sorted(pair_hits.items(), key=lambda kv: kv[1], reverse=True)[:20]
        ],
    }


# =========================
# Audit metrics
# =========================
def percentile_bin_labels() -> list[str]:
    return ["P00-P05", "P05-P25", "P25-P50", "P50-P75", "P75-P95", "P95-P100"]


def global_score_bins(original: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    scored_original = pd.to_numeric(original["score"], errors="coerce")
    valid_values = scored_original.dropna()
    if valid_values.empty:
        return pd.DataFrame()

    cut_values = valid_values.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_numpy()

    def label_series(s: pd.Series) -> pd.Series:
        out = pd.Series("MISSING_SCORE", index=s.index, dtype="object")
        valid = s.notna()
        ranks = s.loc[valid]
        labels = np.select(
            [ranks <= cut_values[0], ranks <= cut_values[1], ranks <= cut_values[2], ranks <= cut_values[3], ranks <= cut_values[4]],
            percentile_bin_labels()[:5],
            default=percentile_bin_labels()[5],
        )
        out.loc[valid] = labels
        return out

    orig_bins = label_series(scored_original)
    final_score = pd.to_numeric(final["score"], errors="coerce")
    final_bins = label_series(final_score)
    order = ["MISSING_SCORE"] + percentile_bin_labels()
    rows = []
    for b in order:
        o = int((orig_bins == b).sum())
        f = int((final_bins == b).sum())
        rows.append({
            "scope": "global",
            "score_bin": b,
            "original_n": o,
            "original_pct": round(o / max(len(original), 1) * 100, 3),
            "final_n": f,
            "final_pct": round(f / max(len(final), 1) * 100, 3),
            "final_minus_original_pct_points": round(f / max(len(final), 1) * 100 - o / max(len(original), 1) * 100, 3),
        })
    return pd.DataFrame(rows)


def source_score_distribution(original: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sorted(set(original["source"]) | set(final["source"])):
        o = original[original["source"] == source]
        f = final[final["source"] == source]
        # Audit uses the source-local rank of the ORIGINAL distribution.
        s = pd.to_numeric(o["score"], errors="coerce")
        valid_idx = s.notna()
        ranks = pd.Series(index=s.index, dtype="object")
        if valid_idx.any():
            pct = s.loc[valid_idx].rank(method="average", pct=True)
            labels = np.select(
                [pct <= 0.05, pct <= 0.25, pct <= 0.50, pct <= 0.75, pct <= 0.95],
                percentile_bin_labels()[:5],
                default=percentile_bin_labels()[5],
            )
            ranks.loc[valid_idx] = labels
        ranks.loc[~valid_idx] = "MISSING_SCORE"

        # Apply the original source thresholds to final using raw score cutpoints.
        valid_values = s.dropna()
        if valid_values.empty:
            f_ranks = pd.Series("MISSING_SCORE", index=f.index, dtype="object")
        else:
            cuts = valid_values.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_numpy()
            fs = pd.to_numeric(f["score"], errors="coerce")
            f_ranks = pd.Series("MISSING_SCORE", index=f.index, dtype="object")
            fv = fs.notna()
            if fv.any():
                fr = fs.loc[fv]
                labels = np.select(
                    [fr <= cuts[0], fr <= cuts[1], fr <= cuts[2], fr <= cuts[3], fr <= cuts[4]],
                    percentile_bin_labels()[:5],
                    default=percentile_bin_labels()[5],
                )
                f_ranks.loc[fv] = labels

        for b in ["MISSING_SCORE"] + percentile_bin_labels():
            on = int((ranks == b).sum())
            fn = int((f_ranks == b).sum())
            rows.append({
                "source": source,
                "score_bin": b,
                "original_n": on,
                "original_pct_within_source": round(on / max(len(o), 1) * 100, 3),
                "final_n": fn,
                "final_pct_within_source": round(fn / max(len(f), 1) * 100, 3),
            })
    return pd.DataFrame(rows)


def task_distribution(original: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sorted(set(original["source"]) | set(final["source"])):
        o = original[original["source"] == source]
        f = final[final["source"] == source]
        for task in TASK_LABELS:
            on = int((o["_task_bucket"] == task).sum())
            fn = int((f["_task_bucket"] == task).sum())
            rows.append({
                "source": source,
                "task_bucket": task,
                "original_n": on,
                "original_pct_within_source": round(on / max(len(o), 1) * 100, 3),
                "final_n": fn,
                "final_pct_within_source": round(fn / max(len(f), 1) * 100, 3),
            })
    return pd.DataFrame(rows)


def conversation_metrics(messages: list[dict[str, object]]) -> dict[str, Any]:
    all_parts = [message_to_audit_text(m) for m in messages]
    prompt_parts = [
        message_to_audit_text(m)
        for m in messages
        if m.get("role") in {"system", "user"}
    ]
    answer_parts = [
        message_to_audit_text(m)
        for m in messages
        if m.get("role") == "assistant"
    ]
    all_text = "\n".join(all_parts)
    prompt_text = "\n".join(prompt_parts)
    answer_text = "\n".join(answer_parts)

    letters = len(_LETTER_RE.findall(all_text))
    persian_chars = len(_PERSIAN_RE.findall(all_text))
    persian_ratio = persian_chars / letters if letters else 0.0
    return {
        "message_count": len(messages),
        "user_message_count": sum(1 for m in messages if m.get("role") == "user"),
        "assistant_message_count": sum(1 for m in messages if m.get("role") == "assistant"),
        "has_system_message": int(any(m.get("role") == "system" for m in messages)),
        "is_multiturn": int(sum(1 for m in messages if m.get("role") == "user") >= 2),
        "has_tool_calls": int(any(bool(m.get("tool_calls")) for m in messages)),
        "has_function_call": int(any(bool(m.get("function_call")) for m in messages)),
        "tool_call_message_count": sum(1 for m in messages if m.get("tool_calls")),
        "function_call_message_count": sum(1 for m in messages if m.get("function_call")),
        "persian_char_ratio": persian_ratio,
        "prompt_text": prompt_text,
        "answer_text": answer_text,
    }


def approximate_token_count(text: str) -> int:
    # Deterministic rough proxy, intentionally labeled approximate.
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def load_optional_tokenizer():
    if not TOKENIZER_NAME:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load TOKENIZER_NAME={TOKENIZER_NAME!r}. "
            "Set TOKENIZER_NAME=None for approximate audit instead."
        ) from exc


def add_conversation_audit_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = df.copy()
    metrics = [conversation_metrics(x) for x in work["_canonical_messages"]]
    metric_df = pd.DataFrame(metrics, index=work.index)
    work = pd.concat([work, metric_df.drop(columns=["prompt_text", "answer_text"])], axis=1)

    tokenizer = load_optional_tokenizer()
    method = "approximate_regex_token_count"
    total_tokens = []
    prompt_tokens = []
    answer_tokens = []
    for metric in metrics:
        p = metric["prompt_text"]
        a = metric["answer_text"]
        if tokenizer is None:
            pt = approximate_token_count(p)
            at = approximate_token_count(a)
        else:
            pt = len(tokenizer.encode(p, add_special_tokens=False))
            at = len(tokenizer.encode(a, add_special_tokens=False))
            method = f"exact:{TOKENIZER_NAME}"
        prompt_tokens.append(pt)
        answer_tokens.append(at)
        total_tokens.append(pt + at)

    work["prompt_tokens"] = prompt_tokens
    work["answer_tokens"] = answer_tokens
    work["prompt_answer_tokens"] = total_tokens
    return work, method


def describe_series(s: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(x.size),
        "mean": round(float(x.mean()), 4),
        "median": round(float(x.median()), 4),
        "p95": round(float(x.quantile(0.95)), 4),
    }


def build_token_audit(original: pd.DataFrame, final: pd.DataFrame, token_method: str) -> pd.DataFrame:
    rows = []
    for scope, source, frame in [("global", "__ALL__", final)]:
        rows.extend([
            {"scope": scope, "source": source, "metric": "prompt_answer_tokens", **describe_series(frame["prompt_answer_tokens"])},
            {"scope": scope, "source": source, "metric": "prompt_tokens", **describe_series(frame["prompt_tokens"])},
            {"scope": scope, "source": source, "metric": "answer_tokens", **describe_series(frame["answer_tokens"])},
        ])
    for source, frame in final.groupby("source", sort=True):
        rows.extend([
            {"scope": "source", "source": source, "metric": "prompt_answer_tokens", **describe_series(frame["prompt_answer_tokens"])},
            {"scope": "source", "source": source, "metric": "prompt_tokens", **describe_series(frame["prompt_tokens"])},
            {"scope": "source", "source": source, "metric": "answer_tokens", **describe_series(frame["answer_tokens"])},
        ])
    audit = pd.DataFrame(rows)
    audit["token_count_method"] = token_method
    return audit


def build_source_audit(original: pd.DataFrame, valid_input: pd.DataFrame, exact: pd.DataFrame, near: pd.DataFrame, final: pd.DataFrame, allocation: pd.DataFrame) -> pd.DataFrame:
    sources = sorted(set(original["source"]) | set(final["source"]))
    amap = allocation.set_index("source")
    rows = []
    for source in sources:
        o = original[original["source"] == source]
        v = valid_input[valid_input["source"] == source]
        e = exact[exact["source"] == source]
        n = near[near["source"] == source]
        f = final[final["source"] == source]
        score_vals = pd.to_numeric(f["score"], errors="coerce")
        row = {
            "source": source,
            "original_n": len(o),
            "invalid_message_structure_n": len(o) - len(v),
            "missing_score_original_n": int(pd.to_numeric(o["score"], errors="coerce").isna().sum()),
            "after_exact_dedup_n": len(e),
            "exact_duplicates_removed_n": len(v) - len(e),
            "after_near_dedup_n": len(n),
            "near_duplicates_removed_n": len(e) - len(n),
            "available_n": len(n),
            "target_quota_n": int(amap.loc[source, "quota"]) if source in amap.index else 0,
            "final_n": len(f),
            "final_missing_score_n": int(score_vals.isna().sum()),
            "final_missing_score_pct": round(score_vals.isna().mean() * 100, 3) if len(f) else 0.0,
            "final_multiturn_pct": round(float(f["is_multiturn"].mean()) * 100, 3) if len(f) else 0.0,
            "final_system_message_pct": round(float(f["has_system_message"].mean()) * 100, 3) if len(f) else 0.0,
            "final_tool_call_pct": round(float(f["has_tool_calls"].mean()) * 100, 3) if len(f) else 0.0,
            "final_function_call_pct": round(float(f["has_function_call"].mean()) * 100, 3) if len(f) else 0.0,
            "final_assistant_message_mean": round(float(f["assistant_message_count"].mean()), 4) if len(f) else 0.0,
            "final_assistant_message_median": round(float(f["assistant_message_count"].median()), 4) if len(f) else 0.0,
            "final_assistant_message_p95": round(float(f["assistant_message_count"].quantile(0.95)), 4) if len(f) else 0.0,
            "final_persian_char_ratio_mean": round(float(f["persian_char_ratio"].mean()), 4) if len(f) else 0.0,
            "final_persian_char_ratio_median": round(float(f["persian_char_ratio"].median()), 4) if len(f) else 0.0,
        }
        if source in amap.index:
            row["source_priority"] = float(amap.loc[source, "source_priority"])
            row["scored_n_for_priority"] = int(amap.loc[source, "scored_n"])
            row["missing_score_n_for_priority"] = int(amap.loc[source, "missing_score_n"])
            row["quality_evidence"] = str(amap.loc[source, "quality_evidence"])
        else:
            row["source_priority"] = np.nan
            row["scored_n_for_priority"] = 0
            row["missing_score_n_for_priority"] = 0
            row["quality_evidence"] = "unavailable"
        row["final_score_mean"] = round(float(score_vals.mean()), 6) if score_vals.notna().any() else np.nan
        row["final_score_median"] = round(float(score_vals.median()), 6) if score_vals.notna().any() else np.nan
        row["final_score_p95"] = round(float(score_vals.quantile(0.95)), 6) if score_vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["final_n", "source"], ascending=[False, True], kind="stable")


# =========================
# Main
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, split=SPLIT)
    df = dataset.to_pandas()

    required = {"messages", "source", "score", "original_index"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing expected columns: {sorted(missing)}")

    print(f"Loaded: {len(df):,} rows")
    df["source"] = df["source"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    original_n = len(df)

    print("[0/7] Inspecting and validating ...")
    inspect_message_schema(df)
    df_validity = df.copy()
    df_validity["_canonical_messages"] = df_validity["messages"].map(canonicalize_messages)
    valid_input = df_validity[df_validity["_canonical_messages"].map(has_sft_structure)].copy()
    invalid_structure_n = len(df) - len(valid_input)
    missing_score_original_n = int(df["score"].isna().sum())
    print(f"  invalid message structure: {invalid_structure_n:,}")
    print(f"  missing/non-numeric score: {missing_score_original_n:,} ({missing_score_original_n/max(original_n,1)*100:.2f}%)")

    print("[1/7] Exact deduplication ...")
    exact_before = len(valid_input)
    # Measure exact duplicate groups before removing them.
    key_counts = valid_input.groupby(valid_input["_canonical_messages"].map(canonical_key)).size()
    duplicate_groups = int((key_counts > 1).sum())
    df_exact = exact_dedup(df)
    print(f"  after exact dedup: {len(df_exact):,}")

    print("[2/7] Near-duplicate detection ...")
    df_near = None
    chosen_threshold = None
    near_stats = {}
    for threshold in NEAR_DUP_THRESHOLDS:
        candidate, stats = near_dedup(df_exact, threshold)
        print(
            f"  threshold={threshold:.2f} -> {len(candidate):,} rows; "
            f"LSH candidates={stats['candidate_matches']:,}; "
            f"cross-source candidates={stats['candidate_cross_source_matches']:,}; "
            f"confirmed removed={stats['confirmed_near_duplicate_rows_removed']:,}"
        )
        # We need at least the target after near dedup. Prefer the first threshold that permits target.
        if len(candidate) >= TARGET_N:
            df_near = candidate
            chosen_threshold = threshold
            near_stats = stats
            break
    if df_near is None:
        df_near = df_exact.copy()
        chosen_threshold = None
        near_stats = {
            "candidate_matches": 0,
            "candidate_cross_source_matches": 0,
            "confirmed_near_duplicate_rows_removed": 0,
            "confirmed_cross_source_near_duplicate_rows_removed": 0,
        }
        print("  WARNING: no threshold retained >= target; using exact-deduplicated data.")
    print(f"  selected threshold: {'none' if chosen_threshold is None else f'{chosen_threshold:.2f}'}")
    print(f"  after near dedup: {len(df_near):,}")

    print("[2.5/7] Independent near-duplicate sanity audit ...")
    near_sanity_audit = independent_near_duplicate_sanity_audit(df_exact)
    if near_sanity_audit.get("enabled"):
        print(
            "  sample={:,}; global best p95/p99/max={} / {} / {}".format(
                int(near_sanity_audit["sample_n"]),
                near_sanity_audit["global_best_similarity"]["p95"],
                near_sanity_audit["global_best_similarity"]["p99"],
                near_sanity_audit["global_best_similarity"]["max"],
            )
        )
        print(
            "  same-source max={}; cross-source max={}".format(
                near_sanity_audit["same_source_best_similarity"]["max"],
                near_sanity_audit["cross_source_best_similarity"]["max"],
            )
        )
        print(
            "  unique sample pairs >=0.85: {:,}".format(
                int(near_sanity_audit["unique_sample_pairs_ge_0.85"])
            )
        )
    else:
        print(f"  sanity audit skipped: {near_sanity_audit.get("reason")}")

    print("[3/7] Quality representation + task labels ...")
    df_quality, quality_meta = compute_quality_scores(df_near)
    df_quality = add_task_buckets(df_quality)
    df_quality = assign_stratification_buckets(df_quality)

    source_quality = build_source_quality_table(df_quality)
    allocation = quality_aware_source_allocation(source_quality, TARGET_N)
    print("  source quotas:")
    for _, row in allocation.iterrows():
        print(
            f"    {row['source']}: {int(row['quota']):,} "
            f"(priority={row['source_priority']:.4f}, available={int(row['available_n']):,}, "
            f"missing-score={row['missing_score_pct']:.1f}%)"
        )

    print("[4/7] Stratified score×task sampling with quality-weighted randomness ...")
    rng = np.random.default_rng(SEED)
    pieces: list[pd.DataFrame] = []
    quotas = allocation.set_index("source")["quota"].astype(int).to_dict()

    for source, quota in quotas.items():
        source_df = df_quality[df_quality["source"] == source]
        pieces.append(stratified_source_sample(source_df, int(quota), rng))

    final = pd.concat(pieces, ignore_index=True)
    if len(final) != TARGET_N:
        raise RuntimeError(f"Final dataset has {len(final):,} rows; expected {TARGET_N:,}.")

    # Shuffle final ordering without changing membership.
    final = final.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    final["id"] = "maux:" + final["original_index"].astype(str)

    print("[5/7] Computing final audit metrics ...")
    final, token_method = add_conversation_audit_metrics(final)

    # Exact duplicate check on final canonical keys.
    final_duplicate_rows = int(final["_dedup_key"].duplicated(keep=False).sum())
    if final_duplicate_rows:
        raise RuntimeError(f"Final dataset contains {final_duplicate_rows:,} exact duplicate rows.")

    # Cross-source exact duplicate groups BEFORE exact dedup.
    keyed = valid_input.assign(_dedup_key=valid_input["_canonical_messages"].map(canonical_key))
    cross_source_exact_groups = int(
        keyed.groupby("_dedup_key")["source"].nunique().gt(1).sum()
    )

    # Original-data audit task labels use the same heuristic classifier and the
    # same canonical message parsing used for the final dataset.
    original_audit = df_validity.copy()
    original_audit = add_task_buckets(original_audit)

    score_audit = pd.concat([
        global_score_bins(df, final),
        source_score_distribution(df, final),
    ], ignore_index=True)
    task_audit = task_distribution(original_audit, final)
    task_multilabel_audit = build_task_multilabel_audit(original_audit, final)
    token_audit = build_token_audit(df, final, token_method)
    source_audit = build_source_audit(df, valid_input, df_exact, df_near, final, allocation)

    score_values_final = pd.to_numeric(final["score"], errors="coerce")
    missing_score_final_n = int(score_values_final.isna().sum())
    original_missing_fraction = missing_score_original_n / max(original_n, 1)
    final_missing_fraction = missing_score_final_n / max(TARGET_N, 1)
    if final_missing_fraction > original_missing_fraction + 1e-12:
        raise RuntimeError(
            "Final missing-score fraction exceeds original dataset fraction: "
            f"final={final_missing_fraction:.6f}, original={original_missing_fraction:.6f}."
        )

    # Optional missing-score cap is OFF by default. If enabled, verify it.
    if MAX_MISSING_SCORE_FRACTION is not None:
        max_allowed = int(np.floor(TARGET_N * MAX_MISSING_SCORE_FRACTION))
        if missing_score_final_n > max_allowed:
            raise RuntimeError(
                f"Final missing-score rows={missing_score_final_n:,} exceed configured "
                f"cap={max_allowed:,}. Raise the cap or change missing-score allocation."
            )

    # Global metrics A-L.
    stats = {
        "original_rows": int(original_n),
        "valid_message_rows": int(len(valid_input)),
        "invalid_message_structure_rows": int(invalid_structure_n),
        "original_missing_score_rows": int(missing_score_original_n),
        "original_missing_score_pct": round(missing_score_original_n / max(original_n, 1) * 100, 4),
        "after_exact_dedup_rows": int(len(df_exact)),
        "after_near_dedup_rows": int(len(df_near)),
        "final_rows": int(len(final)),
        "exact_duplicates_removed_rows": int(exact_before - len(df_exact)),
        "exact_duplicate_groups": duplicate_groups,
        "cross_source_exact_duplicate_groups": cross_source_exact_groups,
        "near_duplicates_removed_rows": int(len(df_exact) - len(df_near)),
        "near_duplicate_candidate_count": int(near_stats.get("candidate_matches", 0)),
        "near_duplicate_cross_source_candidate_count": int(near_stats.get("candidate_cross_source_matches", 0)),
        "near_duplicate_confirmed_cross_source_removed_rows": int(near_stats.get("confirmed_cross_source_near_duplicate_rows_removed", 0)),
        "final_exact_duplicate_rows": final_duplicate_rows,
        "final_missing_score_rows": missing_score_final_n,
        "final_missing_score_pct": round(missing_score_final_n / max(TARGET_N, 1) * 100, 4),
        "final_score_mean": round(float(score_values_final.mean()), 6) if score_values_final.notna().any() else None,
        "final_score_median": round(float(score_values_final.median()), 6) if score_values_final.notna().any() else None,
        "final_score_p95": round(float(score_values_final.quantile(0.95)), 6) if score_values_final.notna().any() else None,
        "final_multiturn_pct": round(float(final["is_multiturn"].mean()) * 100, 4),
        "final_system_message_pct": round(float(final["has_system_message"].mean()) * 100, 4),
        "final_assistant_message_mean": round(float(final["assistant_message_count"].mean()), 4),
        "final_assistant_message_median": round(float(final["assistant_message_count"].median()), 4),
        "final_assistant_message_p95": round(float(final["assistant_message_count"].quantile(0.95)), 4),
        "final_persian_char_ratio_mean": round(float(final["persian_char_ratio"].mean()), 4),
        "final_persian_char_ratio_median": round(float(final["persian_char_ratio"].median()), 4),
        "token_count_method": token_method,
        "missing_score_fraction_guard": {
            "original": round(original_missing_fraction, 6),
            "final": round(final_missing_fraction, 6),
            "passed": bool(final_missing_fraction <= original_missing_fraction + 1e-12),
        },
        "near_duplicate_sanity_audit": near_sanity_audit,
    }

    summary = {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "seed": SEED,
        "requested_target": TARGET_N,
        "selected_near_dedup_threshold": chosen_threshold,
        "near_dedup": {
            "num_perm": NUM_PERM,
            "shingle_size": SHINGLE_SIZE,
            "threshold_schedule": list(NEAR_DUP_THRESHOLDS),
            "lsh_min_threshold": LSH_MIN_THRESHOLD,
            **near_stats,
        },
        "source_allocation": {
            "method": "quality-aware priority with source floor/cap; source size is availability only",
            "min_source_fraction": MIN_SOURCE_FRACTION,
            "max_source_fraction": MAX_SOURCE_FRACTION,
            "source_priority_power": SOURCE_PRIORITY_POWER,
            "quality_top_fraction": QUALITY_TOP_FRACTION,
            "global_score_weight": GLOBAL_SCORE_WEIGHT,
            "source_rank_weight": SOURCE_RANK_WEIGHT,
            "unscored_source_priority": "global_mean_prior",
            "quality_evidence_field": "quality_evidence",
        },
        "stratification": {
            "score_buckets": ["P00-P05", "P05-P25", "P25-P50", "P50-P75", "P75-P95", "P95-P100"],
            "missing_score_bucket": "MISSING_SCORE",
            "task_buckets": list(TASK_LABELS),
            "quality_sampling_temperature": QUALITY_SAMPLING_TEMPERATURE,
            "method": "source quota -> score×task proportional buckets -> quality-weighted random sampling inside scored buckets; random sampling inside missing-score bucket",
            "tool_payload_in_audit": True,
            "task_classifier": "multi-label internal audit + single-label primary_task precedence for selection",
            "missing_score_max_fraction": MAX_MISSING_SCORE_FRACTION,
        },
        "quality_normalization": quality_meta,
        "A_to_L_audit": stats,
        "audit_outputs": {
            "task_multilabel": str(OUTPUT_TASK_MULTILABEL_AUDIT),
        },
        "source_quotas": {str(r["source"]): int(r["quota"]) for _, r in allocation.iterrows()},
        "note": "Mauxi-SFT-Persian was not added separately because it is already a source inside Maux-Persian-SFT-30k. Task buckets are heuristic audit labels, not native dataset metadata.",
    }

    print("[6/7] Saving outputs ...")
    final_output = final[["id", "messages", "source", "score", "original_index"]].copy()
    hf_final = Dataset.from_pandas(final_output, preserve_index=False)
    hf_final.to_parquet(str(OUTPUT_PARQUET))

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for record in final_output.to_dict(orient="records"):
            f.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")

    source_audit.to_csv(OUTPUT_SOURCE_AUDIT, index=False, encoding="utf-8-sig")
    score_audit.to_csv(OUTPUT_SCORE_AUDIT, index=False, encoding="utf-8-sig")
    task_audit.to_csv(OUTPUT_TASK_AUDIT, index=False, encoding="utf-8-sig")
    task_multilabel_audit.to_csv(OUTPUT_TASK_MULTILABEL_AUDIT, index=False, encoding="utf-8-sig")
    token_audit.to_csv(OUTPUT_TOKEN_AUDIT, index=False, encoding="utf-8-sig")
    OUTPUT_AUDIT_SUMMARY.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[7/7] Done.")
    print(f"  Final rows              : {len(final_output):,}")
    print(f"  Final missing score     : {missing_score_final_n:,} ({missing_score_final_n/TARGET_N*100:.2f}%)")
    print(f"  Exact duplicates final  : {final_duplicate_rows:,}")
    print(f"  Near candidates         : {near_stats.get('candidate_matches', 0):,}")
    print(f"  Near removed            : {len(df_exact)-len(df_near):,}")
    print(f"  Near sanity unique pairs >=0.85: {near_sanity_audit.get('unique_sample_pairs_ge_0.85', 0):,}" )
    print(f"  Missing-score guard      : {'PASS' if final_missing_fraction <= original_missing_fraction + 1e-12 else 'FAIL'}")
    print(f"  Stable ID               : maux:<original_index>")
    print(f"  Token audit             : {token_method}")
    print(f"  Output directory        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
