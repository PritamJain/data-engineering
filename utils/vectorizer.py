"""
Vectorizer — semantic similarity matching using sentence-transformers.

Why vectorization improves MDM matching:
  Rule-based matching (exact, phonetic) misses records where the same entity
  is described differently across systems — "St. Mary's Hosp." vs "Saint Mary
  Hospital", or a record where NPI is missing but everything else matches.
  Embedding-based similarity catches these soft duplicates.

Architecture:
  1. Each record → text string (concatenate non-null field values)
  2. All records embedded using sentence-transformers (all-MiniLM-L6-v2, 80 MB)
  3. Cosine similarity computed between all pairs (batched for memory efficiency)
  4. Pairs above threshold returned as "semantic match candidates"
  5. Compared against rule-based results to surface rule gaps

Fallback:
  If sentence-transformers is not installed, falls back to TF-IDF cosine similarity
  (sklearn). If sklearn is also missing, returns empty results gracefully.

Memory note:
  For large datasets, pairwise cosine on N×N embeddings is O(N²).
  We cap at 5,000 records for the full pairwise check. For larger datasets we
  use MiniBatchKMeans clustering to find candidate clusters first.
"""

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_MODEL_NAME   = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality
_MAX_PAIRWISE = 5_000                 # above this, use cluster-based approach
_CACHE: dict  = {}                    # in-process embedding cache keyed by df hash


def _df_hash(df: pd.DataFrame) -> str:
    """Quick hash of the dataframe to cache embeddings."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


def _records_to_text(df: pd.DataFrame, cols: Optional[list[str]] = None) -> list[str]:
    """
    Convert each row to a single text string for embedding.
    Uses all string/object columns, or the specified subset.
    Joins as "field: value" pairs so the embedding captures field semantics.
    """
    use_cols = cols if cols else [
        c for c in df.columns
        if df[c].dtype == object or str(df[c].dtype).startswith("string")
    ]
    texts = []
    for _, row in df[use_cols].iterrows():
        parts = []
        for col in use_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip() not in ("", "nan", "None"):
                parts.append(f"{col}: {str(val).strip()}")
        texts.append(" | ".join(parts) if parts else "empty record")
    return texts


def _embed_st(texts: list[str]) -> np.ndarray:
    """Embed texts using sentence-transformers."""
    model = SentenceTransformer(_MODEL_NAME)
    return model.encode(texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True)


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    """Fallback: TF-IDF embeddings (no download required)."""
    vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
    matrix = vec.fit_transform(texts).toarray()
    # L2 normalise for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def embed_records(df: pd.DataFrame,
                  cols: Optional[list[str]] = None) -> Optional[np.ndarray]:
    """
    Return an (N, D) embedding matrix for the rows of df.
    Returns None if no embedding library is available.
    """
    if not _HAS_ST and not _HAS_SKLEARN:
        return None

    key = _df_hash(df) + str(cols)
    if key in _CACHE:
        return _CACHE[key]

    texts = _records_to_text(df, cols)
    embeddings = _embed_st(texts) if _HAS_ST else _embed_tfidf(texts)
    _CACHE[key] = embeddings
    return embeddings


def find_semantic_matches(
    df: pd.DataFrame,
    threshold: float = 0.85,
    cols: Optional[list[str]] = None,
    max_pairs: int = 10_000,
) -> dict:
    """
    Find semantically similar record pairs using embedding cosine similarity.

    Returns:
        {
          "matching_profiles": int,
          "matching_pairs":    int,
          "largest_cluster":   int,
          "threshold":         float,
          "engine":            "sentence-transformers" | "tfidf" | "unavailable",
          "sample_pairs":      list of {idx_a, idx_b, score, preview_a, preview_b}
        }
    """
    empty = {
        "matching_profiles": 0, "matching_pairs": 0, "largest_cluster": 0,
        "threshold": threshold,  "engine": "unavailable", "sample_pairs": [],
    }
    if not _HAS_ST and not _HAS_SKLEARN:
        return empty

    n = len(df)
    if n < 2:
        return empty

    engine = "sentence-transformers" if _HAS_ST else "tfidf"
    embeddings = embed_records(df, cols)
    if embeddings is None:
        return empty

    matched_indices: set[int] = set()
    pair_list: list[dict]     = []

    if n <= _MAX_PAIRWISE:
        # Full pairwise cosine (vectorised — no Python loop)
        # embeddings are already L2-normalised so dot product = cosine
        sim_matrix = embeddings @ embeddings.T
        rows_idx, cols_idx = np.where(
            (sim_matrix >= threshold) & (np.triu(np.ones((n, n)), k=1) > 0)
        )
        for a, b in zip(rows_idx.tolist(), cols_idx.tolist()):
            score = float(sim_matrix[a, b])
            matched_indices.add(a)
            matched_indices.add(b)
            if len(pair_list) < max_pairs:
                pair_list.append({
                    "idx_a":     int(a),
                    "idx_b":     int(b),
                    "score":     round(score, 4),
                    "preview_a": _record_preview(df, a),
                    "preview_b": _record_preview(df, b),
                })
    else:
        # Cluster-based approximation for large datasets
        if not _HAS_SKLEARN:
            return {**empty, "engine": engine}
        n_clusters = max(10, n // 50)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                 batch_size=1024)
        labels = kmeans.fit_predict(embeddings)
        for cluster_id in range(n_clusters):
            idxs = np.where(labels == cluster_id)[0]
            if len(idxs) < 2:
                continue
            sub_emb = embeddings[idxs]
            sub_sim = sub_emb @ sub_emb.T
            r, c_ = np.where(
                (sub_sim >= threshold) & (np.triu(np.ones(sub_sim.shape), k=1) > 0)
            )
            for ra, rb in zip(r.tolist(), c_.tolist()):
                a, b = int(idxs[ra]), int(idxs[rb])
                score = float(sub_sim[ra, rb])
                matched_indices.add(a)
                matched_indices.add(b)
                if len(pair_list) < max_pairs:
                    pair_list.append({
                        "idx_a": a, "idx_b": b,
                        "score": round(score, 4),
                        "preview_a": _record_preview(df, a),
                        "preview_b": _record_preview(df, b),
                    })

    # Sort sample pairs by score descending
    pair_list.sort(key=lambda x: x["score"], reverse=True)

    # Compute largest cluster using union-find
    largest = _largest_cluster(pair_list, n)

    return {
        "matching_profiles": len(matched_indices),
        "matching_pairs":    len(pair_list),
        "largest_cluster":   largest,
        "threshold":         threshold,
        "engine":            engine,
        "sample_pairs":      pair_list[:50],   # keep top 50 for UI
    }


def compare_with_rules(
    rule_results: dict[str, dict],
    vector_results: dict,
    df: pd.DataFrame,
) -> dict:
    """
    Compare rule-based and embedding-based match pairs to surface gaps.

    Returns:
        {
          "rule_only_profiles":   int  ← caught by rules, not by embeddings
          "vector_only_profiles": int  ← caught by embeddings, not by rules
          "overlap_profiles":     int  ← caught by both (validation)
          "gap_pct":              float ← % of vector matches not covered by rules
          "gap_sample":           list of record previews rules missed
        }
    """
    # Collect all row indices matched by rules
    rule_matched: set[int] = set()
    for uri, c in rule_results.items():
        # We don't have the actual indices from rule simulation, so approximate
        # using matching_profiles count as a fraction
        pass

    # Collect row indices from vector results
    vec_pairs  = vector_results.get("sample_pairs", [])
    vec_idxs   = set()
    for p in vec_pairs:
        vec_idxs.add(p["idx_a"])
        vec_idxs.add(p["idx_b"])

    total_rule_mp = sum(v.get("matching_profiles", 0) for v in rule_results.values())
    total_vec_mp  = vector_results.get("matching_profiles", 0)

    # Approximate overlap (exact overlap needs per-pair indices from rule sim)
    # We estimate: overlap ≈ min(rule, vector) * correlation factor
    overlap_est = min(total_rule_mp, total_vec_mp)
    gap         = max(0, total_vec_mp - overlap_est)
    gap_pct     = round(gap / total_vec_mp * 100, 1) if total_vec_mp else 0

    # Sample pairs only in vector (not in any rule column groups)
    gap_sample = [
        p for p in vec_pairs
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


def _is_gap_pair(pair: dict, rule_results: dict, df: pd.DataFrame) -> bool:
    """
    Heuristic: a pair is a "gap" if the two records share no common non-null
    values in any high-cardinality column (meaning rules wouldn't catch it).
    """
    a, b = pair["idx_a"], pair["idx_b"]
    if a >= len(df) or b >= len(df):
        return False
    row_a = df.iloc[a]
    row_b = df.iloc[b]
    for col in df.columns:
        va = str(row_a[col]).strip()
        vb = str(row_b[col]).strip()
        if va not in ("", "nan", "None") and va == vb:
            return False   # at least one exact match → rules would catch it
    return True


def _record_preview(df: pd.DataFrame, idx: int, max_cols: int = 4) -> str:
    """Return a short human-readable preview of a record."""
    if idx >= len(df):
        return "?"
    row  = df.iloc[idx]
    cols = [c for c in df.columns if df[c].dtype == object][:max_cols]
    parts = []
    for c in cols:
        v = str(row[c]).strip()
        if v not in ("", "nan", "None"):
            parts.append(v)
    return " · ".join(parts[:4]) if parts else f"row {idx}"


def _largest_cluster(pairs: list[dict], n: int) -> int:
    """Union-find to compute the largest connected component from pairs."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for p in pairs:
        union(p["idx_a"], p["idx_b"])

    from collections import Counter
    roots   = [find(i) for i in range(n)]
    counts  = Counter(roots)
    largest = max(counts.values()) if counts else 0
    return int(largest) if largest > 1 else 0


def available() -> bool:
    """Return True if at least one embedding backend is installed."""
    return _HAS_ST or _HAS_SKLEARN


def backend_name() -> str:
    if _HAS_ST:
        return f"sentence-transformers ({_MODEL_NAME})"
    if _HAS_SKLEARN:
        return "TF-IDF (sklearn fallback)"
    return "unavailable"
