"""
Profiling utility — works in Streamlit locally and in Databricks Apps.

Design principle (v2):
  We do NOT pre-classify semantic types via regex — that was the root cause of
  missing NPI, LicenseNumber, Address fields.

  Instead we:
    • Scan the FULL dataset (not 5 rows)
    • Extract 30 distinct real sample values
    • Compute format patterns and length distributions
    • Let the LLM reason about semantics from actual data
"""

import re
from collections import Counter
from typing import Any

import pandas as pd

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

# ── Format pattern detectors (hints only — not classifications) ───────────────
_ALL_DIGITS    = re.compile(r"^\d+$")
_ALPHANUMERIC  = re.compile(r"^[A-Za-z0-9]+$")
_ALPHA_ONLY    = re.compile(r"^[A-Za-z\s\-'.]+$")
_DATE_LIKE     = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}")
_EMAIL_LIKE    = re.compile(r"@[\w.\-]+\.[a-z]{2,}", re.I)
_PHONE_LIKE    = re.compile(r"^\+?[\d\s\-().]{7,15}$")
_ZIP_LIKE      = re.compile(r"^\d{5}(-\d{4})?$")


def _detect_format_hint(values: list[str]) -> str:
    """Return a human-readable format description for the LLM."""
    sample = [str(v).strip() for v in values[:50] if str(v).strip()]
    if not sample:
        return "no samples"

    lengths = [len(v) for v in sample]
    min_l, max_l = min(lengths), max(lengths)
    consistent   = min_l == max_l
    n            = len(sample)

    def pct(k): return k / n

    cnt_digits = sum(1 for v in sample if _ALL_DIGITS.match(v))
    cnt_alpha  = sum(1 for v in sample if _ALPHA_ONLY.match(v))
    cnt_email  = sum(1 for v in sample if _EMAIL_LIKE.search(v))
    cnt_phone  = sum(1 for v in sample if _PHONE_LIKE.match(v))
    cnt_date   = sum(1 for v in sample if _DATE_LIKE.search(v))
    cnt_zip    = sum(1 for v in sample if _ZIP_LIKE.match(v))
    cnt_alnum  = sum(1 for v in sample if _ALPHANUMERIC.match(v))

    if pct(cnt_email) > 0.7:
        return "email address format"
    if pct(cnt_phone) > 0.7:
        return "phone number format"
    if pct(cnt_date)  > 0.7:
        return "date format"
    if pct(cnt_zip)   > 0.7:
        return "US ZIP code format"
    if pct(cnt_digits) > 0.85:
        return (f"all-digit fixed-length ({min_l} digits)"
                if consistent else
                f"all-digit variable-length ({min_l}–{max_l} digits)")
    if pct(cnt_alpha) > 0.85:
        return (f"alphabetic fixed-length ({min_l} chars)"
                if consistent else
                f"alphabetic variable-length ({min_l}–{max_l} chars)")
    if pct(cnt_alnum) > 0.75:
        return (f"alphanumeric fixed-length ({min_l} chars)"
                if consistent else
                f"alphanumeric variable-length ({min_l}–{max_l} chars)")

    # Structural code patterns like "CA-12345" or "A123456"
    patterns     = [re.sub(r"[A-Za-z]", "X", re.sub(r"\d", "N", v)) for v in sample[:15]]
    top_pat, top_cnt = Counter(patterns).most_common(1)[0]
    if top_cnt / min(n, 15) > 0.55 and len(set(patterns)) <= 3:
        return f"structured code, dominant pattern: {top_pat}"

    return f"mixed text ({min_l}–{max_l} chars)"


def _attr_uri_to_col(uri: str) -> str:
    """Extract bare column name from a Reltio attribute URI."""
    return uri.rstrip("/").split("/")[-1]


def _metaphone(value: str) -> str:
    """Double metaphone with 4-char prefix fallback."""
    if _HAS_JELLYFISH:
        try:
            code = jellyfish.metaphone(str(value))
            return code if code else str(value).lower()
        except Exception:
            pass
    return str(value).lower()[:4]


# ── Public API ────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """
    Profile every column across the FULL dataset.

    Sends per column to the LLM:
      dtype, n_rows, n_unique, p_unique, p_missing,
      min/max/mean length (string cols),
      is_numeric_string flag,
      format_hint (human-readable pattern),
      sample_values (up to 30 distinct real values from full data)

    We deliberately do NOT send a pre-guessed semantic type.
    The LLM decides from real data.
    """
    null_thresh = config.get("null_threshold",        0.80)
    card_thresh = config.get("cardinality_threshold", 0.90)
    total_rows  = len(df)
    summary: dict[str, Any] = {}

    for col in df.columns:
        series    = df[col]
        n_null    = int(series.isna().sum())
        p_missing = round(n_null / total_rows, 4) if total_rows else 0.0

        if p_missing > null_thresh:
            continue

        n_unique  = int(series.nunique())
        p_unique  = round(n_unique / total_rows, 4) if total_rows else 0.0
        dtype_str = str(series.dtype)

        if series.dtype == object or str(series.dtype).startswith("string"):
            str_s   = series.dropna().astype(str).str.strip()
            lens    = str_s.str.len()
            min_len = int(lens.min())  if len(lens) else 0
            max_len = int(lens.max())  if len(lens) else 0
            avg_len = round(float(lens.mean()), 1) if len(lens) else 0

            is_numeric_str = bool(
                str_s.str.match(r"^\d+$").sum() / max(len(str_s), 1) > 0.85
            )

            # 30 most-common distinct values from the ENTIRE dataset
            samples = (
                str_s[str_s != ""]
                .value_counts()
                .head(30)
                .index.tolist()
            )
            fmt_hint = _detect_format_hint(samples)

        else:
            min_len = max_len = avg_len = None
            is_numeric_str = False
            samples  = [str(v) for v in series.dropna().unique()[:30]]
            fmt_hint = f"numeric ({dtype_str})"

        summary[col] = {
            "dtype":               dtype_str,
            "n_rows":              total_rows,
            "n_unique":            n_unique,
            "p_unique":            p_unique,
            "p_missing":           p_missing,
            "is_high_cardinality": p_unique >= card_thresh,
            "min_length":          min_len,
            "max_length":          max_len,
            "avg_length":          avg_len,
            "is_numeric_string":   is_numeric_str,
            "format_hint":         fmt_hint,
            "sample_values":       samples[:30],
        }

    return summary


def simulate_match_counts(df: pd.DataFrame, match_rules: dict) -> dict[str, dict]:
    """
    Simulate per-rule match counts across the full df.

    For relevance_based rules with weights/actionThresholds we approximate by
    treating all weighted attributes as exact (conservative lower bound).

    Returns per rule URI:
        matching_profiles  — rows involved in >=1 match
        matching_pairs     — total candidate pairs
        largest_cluster    — biggest duplicate group
        method             — "exact" | "fuzzy+exact" | "weighted_approx" | "unavailable"
    """
    results = {}

    for group in match_rules.get("matchGroups", []):
        uri        = group.get("uri", "")
        rule_key   = "negativeRule" if "negativeRule" in group else "rule"
        rule       = group.get(rule_key, {})
        and_clause = rule.get("and", {})
        rule_type  = group.get("type", "suspect")

        exact_uris = and_clause.get("exact", [])
        fuzzy_uris = and_clause.get("fuzzy", [])

        # For relevance_based, also pull weighted attributes
        weights    = rule.get("weights", [])
        if rule_type == "relevance_based" and weights and not exact_uris:
            exact_uris = [w["attribute"] for w in weights if "attribute" in w]

        exact_cols = [_attr_uri_to_col(u) for u in exact_uris if _attr_uri_to_col(u) in df.columns]
        fuzzy_cols = [_attr_uri_to_col(u) for u in fuzzy_uris if _attr_uri_to_col(u) in df.columns]

        if not exact_cols and not fuzzy_cols:
            results[uri] = {"matching_profiles": 0, "matching_pairs": 0,
                            "largest_cluster": 0, "method": "unavailable — columns not in dataset"}
            continue

        try:
            work   = df.copy()
            method = "exact"

            if fuzzy_cols:
                method = "fuzzy+exact"
                for fc in fuzzy_cols:
                    work[f"__meta_{fc}"] = work[fc].fillna("").astype(str).apply(_metaphone)

            if rule_type == "relevance_based" and weights:
                method = "weighted_approx (lower bound)"

            group_cols = ([f"__meta_{fc}" for fc in fuzzy_cols] + exact_cols) or exact_cols
            avail      = [c for c in group_cols if c in work.columns]
            subset     = work.dropna(subset=avail)
            subset     = subset[subset[avail].ne("").all(axis=1)]

            if subset.empty:
                results[uri] = {"matching_profiles": 0, "matching_pairs": 0,
                                "largest_cluster": 0, "method": method}
                continue

            grp_df     = subset.groupby(avail, dropna=True).size().reset_index(name="__cnt")
            dups       = grp_df[grp_df["__cnt"] > 1]
            results[uri] = {
                "matching_profiles": int(dups["__cnt"].sum()),
                "matching_pairs":    int((dups["__cnt"] * (dups["__cnt"] - 1) / 2).sum()),
                "largest_cluster":   int(dups["__cnt"].max()) if not dups.empty else 0,
                "method":            method,
            }
        except Exception as e:
            results[uri] = {"matching_profiles": 0, "matching_pairs": 0,
                            "largest_cluster": 0, "method": f"error: {e}"}

    return results
