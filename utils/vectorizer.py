"""
Vectorizer v2 — two-stage hybrid pipeline + match evidence extraction.

NEW in v2: extract_match_evidence() + run_evidence_pipeline()
─────────────────────────────────────────────────────────────
After Stage 1 (embedding retrieval) and Stage 2 (Claude adjudication),
we now add a Stage 3:

  Stage 3 — Evidence extraction
    Analyse pairs Claude confirmed as MATCH.
    For every data field, compute:
      • frequency      — how often this field agreed in MATCH pairs
      • match_type     — exact | fuzzy_phonetic | mixed
      • weight_suggestion — 0.0–1.0, used in Pass 2.5 rule generation
      • examples       — sample value pairs that triggered the match

  Stage 4 — Evidence-driven rule generation (in llm.py Pass 2.5)
    Feed the evidence dict to Claude with the semantic analysis.
    Claude generates rules grounded in REAL data patterns, not just schema.

Why this beats schema-only rule generation:
  Schema tells Claude "NPI looks like a strong identifier".
  Evidence tells Claude "NPI matched in 94% of confirmed duplicate pairs".
  The second is fact; the first is inference. Rules from facts are better.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import jellyfish as _jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

_MODEL_NAME   = "all-MiniLM-L6-v2"
_MAX_PAIRWISE = 5_000
_EMBED_CACHE: dict = {}


# ── Text representation ───────────────────────────────────────────────────────

def _record_to_text(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for col in cols:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() not in ("", "nan", "None"):
            parts.append(f"{col}: {str(val).strip()}")
    return " | ".join(parts) if parts else "empty record"


def _records_to_texts(df: pd.DataFrame, cols: Optional[list[str]] = None) -> list[str]:
    use_cols = cols or [
        c for c in df.columns
        if df[c].dtype == object or str(df[c].dtype).startswith("string")
    ]
    return [_record_to_text(row, use_cols) for _, row in df[use_cols].iterrows()]


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# ── Stage 1: Embedding-based candidate retrieval ─────────────────────────────

def _embed_st(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(_MODEL_NAME)
    return model.encode(
        texts, batch_size=32, show_progress_bar=False,
        normalize_embeddings=True,
    )


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    vec    = TfidfVectorizer(max_features=256, sublinear_tf=True)
    matrix = vec.fit_transform(texts).toarray()
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (matrix / norms).astype(np.float32)


def embed_records(df: pd.DataFrame,
                  cols: Optional[list[str]] = None) -> Optional[np.ndarray]:
    if not _HAS_ST and not _HAS_SKLEARN:
        return None
    key = _df_hash(df) + str(cols)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    texts = _records_to_texts(df, cols)
    try:
        embeddings = _embed_st(texts) if _HAS_ST else _embed_tfidf(texts)
    except Exception as e:
        logging.warning(f"Embedding failed ({type(e).__name__}): {e}. Falling back to TF-IDF.")
        if _HAS_SKLEARN:
            try:
                embeddings = _embed_tfidf(texts)
            except Exception:
                return None
        else:
            return None
    embeddings = embeddings.astype(np.float32)
    _EMBED_CACHE[key] = embeddings
    return embeddings


def find_candidate_pairs(
    df: pd.DataFrame,
    threshold: float = 0.78,
    cols: Optional[list[str]] = None,
    max_candidates: int = 1_000,
) -> list[dict]:
    """
    Stage 1: Use embeddings to find candidate duplicate pairs above threshold.
    Returns list of dicts: {idx_a, idx_b, embed_score, preview_a, preview_b}
    """
    embeddings = embed_records(df, cols)
    if embeddings is None:
        return []

    n = len(df)
    pairs: list[dict] = []

    if n <= _MAX_PAIRWISE:
        emb32 = embeddings.astype(np.float32)
        sim   = emb32 @ emb32.T
        rows_i, cols_i = np.where(
            (sim >= threshold) & (np.triu(np.ones((n, n), dtype=np.float32), k=1) > 0)
        )
        for a, b in zip(rows_i.tolist(), cols_i.tolist()):
            pairs.append({
                "idx_a":       int(a),
                "idx_b":       int(b),
                "embed_score": round(float(sim[a, b]), 4),
                "preview_a":   _record_preview(df, int(a)),
                "preview_b":   _record_preview(df, int(b)),
            })
    else:
        if not _HAS_SKLEARN:
            return []
        n_clusters = max(20, n // 100)
        kmeans     = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
        labels     = kmeans.fit_predict(embeddings)
        for cid in range(n_clusters):
            idxs = np.where(labels == cid)[0]
            if len(idxs) < 2:
                continue
            sub     = embeddings[idxs]
            sub_sim = sub @ sub.T
            r, c    = np.where(
                (sub_sim >= threshold) & (np.triu(np.ones(sub_sim.shape), k=1) > 0)
            )
            for ra, rb in zip(r.tolist(), c.tolist()):
                a, b = int(idxs[ra]), int(idxs[rb])
                pairs.append({
                    "idx_a":       a,
                    "idx_b":       b,
                    "embed_score": round(float(sub_sim[ra, rb]), 4),
                    "preview_a":   _record_preview(df, a),
                    "preview_b":   _record_preview(df, b),
                })

    pairs.sort(key=lambda x: x["embed_score"], reverse=True)
    return pairs[:max_candidates]


# ── Stage 2: Claude intelligent reranking ────────────────────────────────────

_RERANK_SYSTEM = """You are a senior Master Data Management (MDM) expert specialising in
healthcare provider (HCP) data deduplication.

You will receive pairs of HCP records and must determine if they represent the same
real-world person.

For each pair output a JSON object with exactly these fields:
  "verdict":    "MATCH" | "SUSPECT" | "NO_MATCH"
  "confidence": 0.0 to 1.0
  "reasoning":  one sentence explaining the key evidence

VERDICT DEFINITIONS:
  MATCH     — High confidence same person. Reltio should auto-merge.
  SUSPECT   — Probable same person but needs data steward review.
  NO_MATCH  — Different people or insufficient evidence to decide.

DOMAIN KNOWLEDGE:
  Names:      Mike=Michael, Dave=David, Raj=Rajesh, Sara=Sarah, etc.
  Dates:      03/15/1968 = 1968-03-15 (same date, different format).
  Specialty:  GI=Gastroenterology, Cards=Cardiology, Peds=Pediatrics.
  State:      Massachusetts=MA, Ohio=OH, California=CA.
  Gender:     Male=M, Female=F.
  NPI:        10-digit globally unique identifier.

Output ONLY the JSON object. No markdown, no prose.""".strip()

_RERANK_USER = """Record A:
{record_a}

Record B:
{record_b}

Are these the same HCP?"""


def _record_to_comparison_text(row: pd.Series) -> str:
    fields = [
        "npi", "first_name", "middle_name", "last_name", "suffix",
        "gender", "dob", "specialty", "license_number", "license_state",
        "dea_number", "practice_name", "address1", "city", "state", "zip",
    ]
    # Use all columns if standard HCP fields aren't present
    use_fields = [f for f in fields if f in row.index] or list(row.index[:16])
    lines = []
    for f in use_fields:
        val = row.get(f, "")
        if pd.notna(val) and str(val).strip() not in ("", "nan", "None"):
            lines.append(f"  {f}: {str(val).strip()}")
    return "\n".join(lines)


def _parse_claude_verdict(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    try:
        parsed = json.loads(raw)
        if "verdict" not in parsed:
            return {"verdict": "SUSPECT", "confidence": 0.5, "reasoning": "Could not parse"}
        return parsed
    except Exception:
        if "MATCH" in raw and "NO_MATCH" not in raw:
            return {"verdict": "MATCH",    "confidence": 0.8, "reasoning": "Parsed from raw text"}
        if "SUSPECT" in raw:
            return {"verdict": "SUSPECT",  "confidence": 0.6, "reasoning": "Parsed from raw text"}
        return {"verdict": "NO_MATCH",     "confidence": 0.5, "reasoning": "Parse failed"}


def rerank_with_claude(
    df: pd.DataFrame,
    candidates: list[dict],
    api_key: str,
    max_pairs: int = 200,
    progress_callback=None,
) -> list[dict]:
    """
    Stage 2: Call Claude on each candidate pair → MATCH / SUSPECT / NO_MATCH.
    Returns candidates list with added claude_verdict, claude_confidence, claude_reasoning.
    """
    if not _HAS_ANTHROPIC or not api_key:
        return candidates

    client  = _anthropic.Anthropic(api_key=api_key)
    results = []
    top     = candidates[:max_pairs]

    for i, pair in enumerate(top):
        if progress_callback:
            progress_callback(i + 1, len(top))

        row_a = df.iloc[pair["idx_a"]]
        row_b = df.iloc[pair["idx_b"]]

        try:
            resp = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 200,
                temperature= 0,
                system     = _RERANK_SYSTEM,
                messages   = [{
                    "role":    "user",
                    "content": _RERANK_USER.format(
                        record_a=_record_to_comparison_text(row_a),
                        record_b=_record_to_comparison_text(row_b),
                    ),
                }],
            )
            verdict = _parse_claude_verdict(resp.content[0].text)
        except Exception as e:
            verdict = {
                "verdict":    "SUSPECT",
                "confidence": pair["embed_score"],
                "reasoning":  f"Claude API error: {str(e)[:60]}",
            }

        results.append({
            **pair,
            "claude_verdict":    verdict.get("verdict",    "SUSPECT"),
            "claude_confidence": verdict.get("confidence", 0.5),
            "claude_reasoning":  verdict.get("reasoning",  ""),
        })

    for pair in candidates[max_pairs:]:
        results.append({
            **pair,
            "claude_verdict":    "SUSPECT",
            "claude_confidence": pair["embed_score"],
            "claude_reasoning":  "Not reranked — beyond max_pairs limit",
        })

    return results


# ── Stage 3: Extract match evidence from confirmed duplicates ─────────────────

def _metaphone_safe(val: str) -> str:
    """Double Metaphone with 4-char prefix fallback."""
    if _HAS_JELLYFISH:
        try:
            code = _jellyfish.metaphone(val)
            return code if code else val[:4]
        except Exception:
            pass
    return val.lower()[:4]


def _norm(val) -> str:
    """Normalise to lowercase stripped string; empty if null."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null", "") else s.lower()


def extract_match_evidence(
    df: pd.DataFrame,
    candidates_with_verdicts: list[dict],
    min_frequency: float = 0.25,
) -> dict:
    """
    Stage 3: Analyse pairs Claude confirmed as MATCH to extract field-level evidence.

    For every data column, computes across all MATCH pairs:
      frequency        — fraction of MATCH pairs where this field agreed
      exact_matches    — count where values were identical (case-insensitive)
      fuzzy_matches    — count where phonetic codes matched
      total_compared   — pairs where both records had this field populated
      match_type       — "exact" | "fuzzy_phonetic" | "mixed"
      examples         — up to 3 sample {a, b, type} dicts
      weight_suggestion — 0.0–1.0 for use in relevance_based rules

    Fields below min_frequency are excluded from the output.

    Returns: dict sorted by frequency descending.
    """
    match_pairs = [p for p in candidates_with_verdicts
                   if p.get("claude_verdict") == "MATCH"]
    if not match_pairs:
        return {}

    data_cols = [c for c in df.columns if not c.startswith("__")]
    field_stats: dict[str, dict] = {}

    for col in data_cols:
        exact_cnt   = 0
        fuzzy_cnt   = 0
        compared    = 0
        examples: list[dict] = []

        for pair in match_pairs:
            a, b = pair.get("idx_a", -1), pair.get("idx_b", -1)
            if a < 0 or b < 0 or a >= len(df) or b >= len(df):
                continue

            va = _norm(df.iloc[a].get(col, ""))
            vb = _norm(df.iloc[b].get(col, ""))

            if not va or not vb:
                continue  # Skip — at least one side is null

            compared += 1

            if va == vb:
                exact_cnt += 1
                if len(examples) < 3:
                    examples.append({
                        "a": str(df.iloc[a].get(col, "")).strip(),
                        "b": str(df.iloc[b].get(col, "")).strip(),
                        "type": "exact",
                    })
            elif _metaphone_safe(va) == _metaphone_safe(vb):
                fuzzy_cnt += 1
                if len(examples) < 3:
                    examples.append({
                        "a": str(df.iloc[a].get(col, "")).strip(),
                        "b": str(df.iloc[b].get(col, "")).strip(),
                        "type": "fuzzy_phonetic",
                    })

        if compared == 0:
            continue

        total_matched = exact_cnt + fuzzy_cnt
        frequency     = round(total_matched / compared, 3)

        if frequency < min_frequency:
            continue

        if exact_cnt > 0 and fuzzy_cnt > 0:
            match_type = "mixed"
        elif fuzzy_cnt > exact_cnt:
            match_type = "fuzzy_phonetic"
        else:
            match_type = "exact"

        # Weight suggestion: frequency * type multiplier
        type_mult = {"exact": 1.0, "mixed": 0.85, "fuzzy_phonetic": 0.70}
        weight    = round(min(frequency * type_mult[match_type], 1.0), 2)

        field_stats[col] = {
            "frequency":          frequency,
            "exact_matches":      exact_cnt,
            "fuzzy_matches":      fuzzy_cnt,
            "total_compared":     compared,
            "match_type":         match_type,
            "examples":           examples,
            "weight_suggestion":  weight,
        }

    # Sort by frequency descending
    return dict(sorted(field_stats.items(),
                        key=lambda x: x[1]["frequency"], reverse=True))


# ── Stage 4 public API: full evidence pipeline ────────────────────────────────

def run_evidence_pipeline(
    df: pd.DataFrame,
    api_key: str,
    threshold: float = 0.78,
    cols: Optional[list[str]] = None,
    max_claude_pairs: int = 200,
    min_evidence_freq: float = 0.25,
    progress_callback=None,
) -> dict:
    """
    Full evidence pipeline for Pass 2.5 rule generation.

    Runs Stage 1 (embedding) + Stage 2 (Claude adjudication) +
    Stage 3 (evidence extraction).

    Returns:
      {
        candidates:       list of adjudicated candidate pairs,
        match_evidence:   field-level evidence dict,
        n_match:          count of MATCH pairs,
        n_suspect:        count of SUSPECT pairs,
        n_no_match:       count of NO_MATCH pairs,
        engine:           backend name string,
      }
    """
    empty = {
        "candidates": [], "match_evidence": {},
        "n_match": 0, "n_suspect": 0, "n_no_match": 0,
        "engine": backend_name(),
    }

    if not _HAS_ST and not _HAS_SKLEARN:
        return empty

    # Stage 1
    candidates = find_candidate_pairs(df, threshold=threshold, cols=cols,
                                      max_candidates=1_000)
    if not candidates:
        return empty

    # Stage 2
    if api_key and _HAS_ANTHROPIC:
        candidates = rerank_with_claude(
            df, candidates, api_key=api_key,
            max_pairs=max_claude_pairs,
            progress_callback=progress_callback,
        )

    # Stage 3
    match_evidence = extract_match_evidence(
        df, candidates, min_frequency=min_evidence_freq
    )

    return {
        "candidates":     candidates,
        "match_evidence": match_evidence,
        "n_match":        sum(1 for p in candidates if p.get("claude_verdict") == "MATCH"),
        "n_suspect":      sum(1 for p in candidates if p.get("claude_verdict") == "SUSPECT"),
        "n_no_match":     sum(1 for p in candidates if p.get("claude_verdict") == "NO_MATCH"),
        "engine":         backend_name(),
    }


# ── Original public API (kept for backward compatibility) ─────────────────────

def find_semantic_matches(
    df: pd.DataFrame,
    threshold: float = 0.78,
    cols: Optional[list[str]] = None,
    api_key: str = "",
    max_claude_pairs: int = 200,
    progress_callback=None,
) -> dict:
    """Original find_semantic_matches — kept for backward compatibility."""
    empty = {
        "matching_profiles": 0, "matching_pairs": 0, "largest_cluster": 0,
        "threshold": threshold,  "engine": "unavailable", "sample_pairs": [],
    }
    if not _HAS_ST and not _HAS_SKLEARN:
        return empty

    engine = ("sentence-transformers + Claude" if (api_key and _HAS_ANTHROPIC) else
              ("sentence-transformers" if _HAS_ST else "TF-IDF"))

    candidates = find_candidate_pairs(df, threshold=threshold, cols=cols, max_candidates=1_000)
    if not candidates:
        return {**empty, "engine": engine}

    if api_key and _HAS_ANTHROPIC and candidates:
        candidates = rerank_with_claude(
            df, candidates, api_key=api_key,
            max_pairs=max_claude_pairs,
            progress_callback=progress_callback,
        )
        active = [p for p in candidates if p.get("claude_verdict", "SUSPECT") != "NO_MATCH"]
    else:
        active = candidates

    matched_idx: set[int] = set()
    for p in active:
        matched_idx.add(p["idx_a"])
        matched_idx.add(p["idx_b"])

    largest = _largest_cluster(active, len(df))

    return {
        "matching_profiles": len(matched_idx),
        "matching_pairs":    len(active),
        "largest_cluster":   largest,
        "threshold":         threshold,
        "engine":            engine,
        "sample_pairs":      active[:50],
        "total_candidates":  len(candidates),
        "claude_reranked":   len([p for p in candidates if "claude_verdict" in p]),
        "claude_matches":    len([p for p in candidates if p.get("claude_verdict") == "MATCH"]),
        "claude_suspects":   len([p for p in candidates if p.get("claude_verdict") == "SUSPECT"]),
        "claude_no_match":   len([p for p in candidates if p.get("claude_verdict") == "NO_MATCH"]),
    }


def compare_with_rules(rule_results: dict, vector_results: dict, df: pd.DataFrame) -> dict:
    total_rule_mp = sum(v.get("matching_profiles", 0) for v in rule_results.values())
    total_vec_mp  = vector_results.get("matching_profiles", 0)
    overlap_est   = min(total_rule_mp, total_vec_mp)
    gap           = max(0, total_vec_mp - overlap_est)
    gap_pct       = round(gap / total_vec_mp * 100, 1) if total_vec_mp else 0

    gap_sample = [
        p for p in vector_results.get("sample_pairs", [])
        if _is_gap_pair(p, rule_results, df)
    ][:10]

    return {
        "rule_matched_profiles":   total_rule_mp,
        "vector_matched_profiles": total_vec_mp,
        "overlap_estimate":        overlap_est,
        "vector_only_estimate":    gap,
        "gap_pct":                 gap_pct,
        "gap_sample":              gap_sample,
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _record_preview(df: pd.DataFrame, idx: int, max_cols: int = 4) -> str:
    if idx >= len(df):
        return "?"
    row  = df.iloc[idx]
    cols = [c for c in df.columns if df[c].dtype == object][:max_cols]
    parts = [str(row[c]).strip() for c in cols
             if str(row.get(c, "")).strip() not in ("", "nan", "None")]
    return " · ".join(parts[:4]) if parts else f"row {idx}"


def _is_gap_pair(pair: dict, rule_results: dict, df: pd.DataFrame) -> bool:
    a, b = pair.get("idx_a", -1), pair.get("idx_b", -1)
    if a < 0 or b < 0 or a >= len(df) or b >= len(df):
        return False
    row_a, row_b = df.iloc[a], df.iloc[b]
    for col in df.columns:
        va = str(row_a.get(col, "")).strip()
        vb = str(row_b.get(col, "")).strip()
        if va not in ("", "nan", "None") and va == vb:
            return False
    return True


def _largest_cluster(pairs: list[dict], n: int) -> int:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for p in pairs:
        if "idx_a" in p and "idx_b" in p:
            union(p["idx_a"], p["idx_b"])

    from collections import Counter
    roots   = [find(i) for i in range(n)]
    counts  = Counter(roots)
    largest = max(counts.values()) if counts else 0
    return int(largest) if largest > 1 else 0


def available() -> bool:
    return _HAS_ST or _HAS_SKLEARN


def backend_name() -> str:
    if _HAS_ST and _HAS_ANTHROPIC:
        return "sentence-transformers + Claude reranking"
    if _HAS_ST:
        return f"sentence-transformers ({_MODEL_NAME})"
    if _HAS_SKLEARN:
        return "TF-IDF (sklearn fallback)"
    return "unavailable"
