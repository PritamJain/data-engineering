"""
Vectorizer — two-stage hybrid matching pipeline.

Stage 1 — Embedding retrieval (fast, local)
  sentence-transformers (all-MiniLM-L6-v2) embeds every record and finds
  candidate pairs above a cosine similarity threshold.
  This reduces the search space from O(N²) to O(N * k) where k is small.
  Falls back to TF-IDF if sentence-transformers is not installed.

Stage 2 — Claude intelligent reranking (accurate, context-aware)
  For each candidate pair found in Stage 1, Claude:
    - Compares every field side by side
    - Understands medical synonyms (GI = Gastroenterology)
    - Understands nicknames (Mike = Michael, Raj = Rajesh)
    - Understands format variations (ISO dates, E.164 phones)
    - Returns MATCH / SUSPECT / NO_MATCH with reasoning
  Claude is ONLY called on the small set of candidates, not all pairs.
  Typical: 50,000 records → embedding finds ~300-500 candidates → Claude
  adjudicates those 300-500 pairs. Efficient and intelligent.

Why this beats either approach alone:
  - Embedding alone: fast but misses semantic synonyms, doesn't reason
  - Claude alone: intelligent but O(N²) API calls at scale = infeasible
  - Combined: embedding does cheap blocking, Claude does smart scoring
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

# Suppress noisy sentence-transformers logs (harmless weight warnings)
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

_MODEL_NAME   = "all-MiniLM-L6-v2"
_MAX_PAIRWISE = 5_000
_EMBED_CACHE: dict = {}


# ── Text representation ───────────────────────────────────────────────────────

def _record_to_text(row: pd.Series, cols: list[str]) -> str:
    """
    Convert a record to a text string for embedding.
    Uses 'field: value' format so the model captures field semantics,
    not just raw values.
    """
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
        texts, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True
    )


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    vec    = TfidfVectorizer(max_features=512, sublinear_tf=True)
    matrix = vec.fit_transform(texts).toarray()
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def embed_records(df: pd.DataFrame,
                  cols: Optional[list[str]] = None) -> Optional[np.ndarray]:
    if not _HAS_ST and not _HAS_SKLEARN:
        return None
    key = _df_hash(df) + str(cols)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    texts      = _records_to_texts(df, cols)
    embeddings = _embed_st(texts) if _HAS_ST else _embed_tfidf(texts)
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
    Returns a list of dicts: {idx_a, idx_b, embed_score, preview_a, preview_b}
    Capped at max_candidates to keep Stage 2 API costs manageable.
    """
    embeddings = embed_records(df, cols)
    if embeddings is None:
        return []

    n = len(df)
    pairs: list[dict] = []

    if n <= _MAX_PAIRWISE:
        sim = embeddings @ embeddings.T
        rows_i, cols_i = np.where(
            (sim >= threshold) & (np.triu(np.ones((n, n)), k=1) > 0)
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
        kmeans     = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                     batch_size=1024)
        labels     = kmeans.fit_predict(embeddings)
        for cid in range(n_clusters):
            idxs = np.where(labels == cid)[0]
            if len(idxs) < 2:
                continue
            sub    = embeddings[idxs]
            sub_sim = sub @ sub.T
            r, c   = np.where(
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

    # Sort by score desc, cap
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
              Use when: NPI matches, OR name+DOB+state all align despite formatting.
  SUSPECT   — Probable same person but needs data steward review.
              Use when: name matches with variation AND at least one other field aligns.
  NO_MATCH  — Different people or insufficient evidence to decide.

DOMAIN KNOWLEDGE you must apply:
  Names:      Mike=Michael, Dave=David, Jim=James, Raj=Rajesh, Sara=Sarah, etc.
              Obrien = O'Brien (dropped apostrophe), Mcdonald = McDonald.
              P. could be any name starting with P — treat as weak evidence only.
  Dates:      03/15/1968 = 1968-03-15 = 15-03-1968 (same date, different format).
  Specialty:  GI=Gastroenterology, Cards=Cardiology, Ortho/Orthopaedics=Orthopedics,
              Family Practice=Family Medicine, Peds=Pediatrics, EM=Emergency Medicine,
              Diabetology=Endocrinology, Haematology=Hematology.
  License:    12345 with state MA = MA-12345 (state prefix stripped).
  State:      Massachusetts=MA, Ohio=OH, California=CA (full name = code).
  Gender:     Male=M, Female=F.
  NPI:        10-digit globally unique identifier. If both records have an NPI and
              they differ by 1-2 digits, likely a typo — treat as supporting evidence.

Output ONLY the JSON object. No markdown, no prose, no explanation outside the JSON.
""".strip()

_RERANK_USER = """Record A:
{record_a}

Record B:
{record_b}

Are these the same HCP?"""


def _record_to_comparison_text(row: pd.Series, label: str) -> str:
    """Format a record for Claude comparison."""
    fields = [
        "npi", "first_name", "middle_name", "last_name", "suffix",
        "gender", "dob", "specialty", "license_number", "license_state",
        "dea_number", "practice_name", "address1", "city", "state", "zip",
    ]
    lines = [f"  {f}: {row.get(f,'')}" for f in fields
             if pd.notna(row.get(f,"")) and str(row.get(f,"")).strip() not in ("","nan","None")]
    return "\n".join(lines)


def _parse_claude_verdict(raw: str) -> dict:
    """Parse Claude's JSON verdict, with fallback."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    try:
        parsed = json.loads(raw)
        if "verdict" not in parsed:
            return {"verdict": "SUSPECT", "confidence": 0.5,
                    "reasoning": "Could not parse verdict"}
        return parsed
    except Exception:
        if "MATCH" in raw and "NO_MATCH" not in raw:
            return {"verdict": "MATCH",    "confidence": 0.8,
                    "reasoning": "Parsed from raw text"}
        if "SUSPECT" in raw:
            return {"verdict": "SUSPECT",  "confidence": 0.6,
                    "reasoning": "Parsed from raw text"}
        return {"verdict": "NO_MATCH",     "confidence": 0.5,
                "reasoning": "Parse failed — defaulting to NO_MATCH"}


def rerank_with_claude(
    df: pd.DataFrame,
    candidates: list[dict],
    api_key: str,
    max_pairs: int = 200,
    progress_callback=None,
) -> list[dict]:
    """
    Stage 2: Call Claude on each candidate pair to get an intelligent verdict.

    Returns the candidates list with added fields:
      claude_verdict:    "MATCH" | "SUSPECT" | "NO_MATCH"
      claude_confidence: 0.0-1.0
      claude_reasoning:  one-sentence explanation

    max_pairs caps API calls (default 200 — ~$0.04 at Sonnet pricing).
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
                        record_a = _record_to_comparison_text(row_a, "A"),
                        record_b = _record_to_comparison_text(row_b, "B"),
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

    # Remaining candidates beyond max_pairs get embed score only
    for pair in candidates[max_pairs:]:
        results.append({
            **pair,
            "claude_verdict":    "SUSPECT",
            "claude_confidence": pair["embed_score"],
            "claude_reasoning":  "Not reranked — beyond max_pairs limit",
        })

    return results


# ── Main public API ───────────────────────────────────────────────────────────

def find_semantic_matches(
    df: pd.DataFrame,
    threshold: float = 0.78,
    cols: Optional[list[str]] = None,
    api_key: str = "",
    max_claude_pairs: int = 200,
    progress_callback=None,
) -> dict:
    """
    Full two-stage pipeline:
      Stage 1: Embedding candidate retrieval
      Stage 2: Claude intelligent reranking (if api_key provided)

    Returns:
      matching_profiles, matching_pairs, largest_cluster,
      threshold, engine, sample_pairs (with claude verdicts if available)
    """
    empty = {
        "matching_profiles": 0, "matching_pairs": 0, "largest_cluster": 0,
        "threshold": threshold,  "engine": "unavailable", "sample_pairs": [],
    }
    if not _HAS_ST and not _HAS_SKLEARN:
        return empty

    engine = "sentence-transformers + Claude" if (api_key and _HAS_ANTHROPIC) else \
             ("sentence-transformers" if _HAS_ST else "TF-IDF")

    # Stage 1: embedding candidates
    candidates = find_candidate_pairs(df, threshold=threshold, cols=cols,
                                      max_candidates=1_000)
    if not candidates:
        return {**empty, "engine": engine}

    # Stage 2: Claude reranking
    if api_key and _HAS_ANTHROPIC and candidates:
        candidates = rerank_with_claude(
            df, candidates, api_key=api_key,
            max_pairs=max_claude_pairs,
            progress_callback=progress_callback,
        )
        # Filter: only count pairs Claude says MATCH or SUSPECT
        active = [p for p in candidates
                  if p.get("claude_verdict", "SUSPECT") != "NO_MATCH"]
    else:
        active = candidates

    # Compute stats
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
        "claude_reranked":   len([p for p in candidates
                                  if "claude_verdict" in p]),
        "claude_matches":    len([p for p in candidates
                                  if p.get("claude_verdict") == "MATCH"]),
        "claude_suspects":   len([p for p in candidates
                                  if p.get("claude_verdict") == "SUSPECT"]),
        "claude_no_match":   len([p for p in candidates
                                  if p.get("claude_verdict") == "NO_MATCH"]),
    }


def compare_with_rules(
    rule_results: dict[str, dict],
    vector_results: dict,
    df: pd.DataFrame,
) -> dict:
    """Compare rule-based and vector-based results to surface gaps."""
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
             if str(row.get(c,"")).strip() not in ("","nan","None")]
    return " · ".join(parts[:4]) if parts else f"row {idx}"


def _is_gap_pair(pair: dict, rule_results: dict, df: pd.DataFrame) -> bool:
    a, b = pair.get("idx_a", -1), pair.get("idx_b", -1)
    if a < 0 or b < 0 or a >= len(df) or b >= len(df):
        return False
    row_a, row_b = df.iloc[a], df.iloc[b]
    for col in df.columns:
        va = str(row_a.get(col,"")).strip()
        vb = str(row_b.get(col,"")).strip()
        if va not in ("","nan","None") and va == vb:
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
        return f"sentence-transformers + Claude reranking"
    if _HAS_ST:
        return f"sentence-transformers ({_MODEL_NAME})"
    if _HAS_SKLEARN:
        return "TF-IDF (sklearn fallback)"
    return "unavailable"
