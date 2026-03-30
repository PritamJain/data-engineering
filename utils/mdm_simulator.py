"""
MDM Simulator — Reltio-like virtual environment.

Simulates the full MDM pipeline on top of generated match rules:

  Stage 1 — Rule application   (exact + phonetic matching per match group)
  Stage 2 — Entity clustering  (Union-Find on matched pairs)
  Stage 3 — Golden records     (survivorship: source priority → most-common fallback)
  Stage 4 — Results packaging  (Entity 360 data model for Streamlit UI)

Output mirrors Reltio's core data model:
  • GoldenRecord  — merged entity with field-level provenance
  • MatchPair     — pair of source records that triggered a rule
  • SimulationResult — summary stats + all golden records + cluster map

Why Union-Find?
  Reltio uses transitive merging — if A=B and B=C then A, B, C are one entity.
  Union-Find is the standard O(n α(n)) algorithm for transitive closure.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchPair:
    idx_a:        int
    idx_b:        int
    rule_uri:     str
    rule_label:   str
    rule_type:    str          # "automatic" | "suspect" | "demoted" (by negativeRule)
    match_fields: list[str]    # field names that triggered the match
    evidence:     dict         # {field: {a_val, b_val, match_type}}


@dataclass
class GoldenField:
    field_name:  str
    value:       str
    source_idx:  int           # original df row index this value came from
    source_name: str           # value of source_col, or "row {idx}"
    reason:      str           # "highest_priority_source" | "most_common" | "only_value"


@dataclass
class GoldenRecord:
    entity_id:    str
    cluster_size: int
    source_idxs:  list[int]   # original df row indices in this cluster
    fields:       list[GoldenField]
    match_type:   str          # "automatic" if any auto pair, else "suspect"
    matched_by:   list[str]   # rule labels that caused merges


@dataclass
class RuleStats:
    label:    str
    type:     str
    pairs:    int
    profiles: int


@dataclass
class SimulationResult:
    # ── Summary ──────────────────────────────────────────────────────────────
    total_input_records:   int
    total_entities:        int      # after dedup
    auto_merged_entities:  int      # clusters where ≥1 pair was auto
    suspect_entities:      int      # clusters where all pairs are suspect
    singleton_entities:    int      # records with no match
    dedup_rate_pct:        float    # (input - entities) / input * 100

    # ── Detailed ─────────────────────────────────────────────────────────────
    golden_records:        list[GoldenRecord]
    match_pairs:           list[MatchPair]
    cluster_map:           dict[int, list[int]]  # root → [record_idxs]

    # ── Per-rule ─────────────────────────────────────────────────────────────
    rule_stats:            dict[str, RuleStats]  # rule_uri → RuleStats


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find (path-compressed, rank-based)
# ─────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


# ─────────────────────────────────────────────────────────────────────────────
# Value helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col_from_uri(uri: str) -> str:
    """Extract bare column name from a Reltio attribute URI."""
    return uri.rstrip("/").split("/")[-1]


def _norm(val) -> str:
    """Normalise a value to lowercase stripped string; empty if null/missing."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null", "") else s.lower()


def _metaphone(val: str) -> str:
    """Double Metaphone with 4-char prefix fallback."""
    if _HAS_JELLYFISH:
        try:
            code = jellyfish.metaphone(val)
            return code if code else val[:4]
        except Exception:
            pass
    return val[:4]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Apply match rules → MatchPairs
# ─────────────────────────────────────────────────────────────────────────────

def _apply_boolean_group(df: pd.DataFrame, group: dict) -> list[MatchPair]:
    """
    Apply one boolean (automatic / suspect) match group.

    Builds a composite grouping key from:
      exact       → lowercased value
      fuzzy       → Double Metaphone code
      exactOrNull → included in key only when both records have a value,
                    treated as a post-filter (doesn't restrict the group key)

    Pairs within the same key bucket are candidates; exactOrNull is
    verified pair-by-pair after grouping.
    """
    rule       = group.get("rule", {})
    and_c      = rule.get("and", {})
    uri        = group.get("uri", "unknown")
    label      = group.get("label", uri.split("/")[-1])
    rtype      = group.get("type", "suspect")

    exact_uris    = and_c.get("exact",       [])
    fuzzy_uris    = and_c.get("fuzzy",       [])
    eon_uris      = and_c.get("exactOrNull", [])

    exact_cols = [_col_from_uri(u) for u in exact_uris    if _col_from_uri(u) in df.columns]
    fuzzy_cols = [_col_from_uri(u) for u in fuzzy_uris    if _col_from_uri(u) in df.columns]
    eon_cols   = [_col_from_uri(u) for u in eon_uris      if _col_from_uri(u) in df.columns]

    if not exact_cols and not fuzzy_cols:
        return []

    work = df.reset_index(drop=True)
    key_cols: list[str] = []

    for col in exact_cols:
        k = f"__ex_{col}"
        work[k] = work[col].apply(_norm)
        key_cols.append(k)

    for col in fuzzy_cols:
        k = f"__fz_{col}"
        work[k] = work[col].apply(lambda v: _metaphone(_norm(v)))
        key_cols.append(k)

    # Only group rows where ALL key columns are non-empty
    mask   = work[key_cols].ne("").all(axis=1)
    subset = work[mask]
    if subset.empty:
        return []

    pairs: list[MatchPair] = []
    for _, grp in subset.groupby(key_cols, dropna=True):
        idxs = grp.index.tolist()
        if len(idxs) < 2:
            continue

        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]

                # exactOrNull post-filter: if both have a value, they must match
                eon_ok = True
                for col in eon_cols:
                    va, vb = _norm(df.iloc[a].get(col)), _norm(df.iloc[b].get(col))
                    if va and vb and va != vb:
                        eon_ok = False
                        break
                if not eon_ok:
                    continue

                # Build evidence dict
                evidence: dict = {}
                match_fields: list[str] = []
                for col in exact_cols:
                    va = str(df.iloc[a].get(col, "")).strip()
                    vb = str(df.iloc[b].get(col, "")).strip()
                    evidence[col] = {"a_val": va, "b_val": vb, "match_type": "exact"}
                    match_fields.append(col)
                for col in fuzzy_cols:
                    va = str(df.iloc[a].get(col, "")).strip()
                    vb = str(df.iloc[b].get(col, "")).strip()
                    evidence[col] = {"a_val": va, "b_val": vb, "match_type": "fuzzy_phonetic"}
                    match_fields.append(col)

                pairs.append(MatchPair(
                    idx_a=a, idx_b=b,
                    rule_uri=uri, rule_label=label, rule_type=rtype,
                    match_fields=match_fields, evidence=evidence,
                ))
    return pairs


def _demote_by_negative_rules(
    pairs: list[MatchPair],
    df: pd.DataFrame,
    match_groups: list[dict],
) -> list[MatchPair]:
    """
    Reltio negative rule semantics: if a negativeRule fires on a pair,
    all 'automatic' verdicts for that pair are demoted to 'suspect'.
    The pair is NOT removed — it still appears in the output as suspect.
    """
    neg_groups = [g for g in match_groups if "negativeRule" in g]
    if not neg_groups:
        return pairs

    neg_cols: list[str] = []
    for ng in neg_groups:
        neg_rule = ng.get("negativeRule", {})
        for uri in neg_rule.get("and", {}).get("notExactSame", []):
            col = _col_from_uri(uri)
            if col in df.columns:
                neg_cols.append(col)

    if not neg_cols:
        return pairs

    result: list[MatchPair] = []
    for p in pairs:
        demoted = False
        if p.rule_type == "automatic":
            for col in neg_cols:
                va = _norm(df.iloc[p.idx_a].get(col))
                vb = _norm(df.iloc[p.idx_b].get(col))
                if va and vb and va != vb:
                    demoted = True
                    break
        result.append(MatchPair(
            idx_a=p.idx_a, idx_b=p.idx_b,
            rule_uri=p.rule_uri, rule_label=p.rule_label,
            rule_type="suspect" if demoted else p.rule_type,
            match_fields=p.match_fields, evidence=p.evidence,
        ))
    return result


def apply_match_rules(df: pd.DataFrame, match_rules: dict) -> list[MatchPair]:
    """
    Stage 1 public API: apply all match groups, return all MatchPairs.
    Negative rules demote automatic → suspect rather than dropping pairs.
    """
    groups    = match_rules.get("matchGroups", [])
    all_pairs: list[MatchPair] = []
    for group in groups:
        if "negativeRule" in group:
            continue
        all_pairs.extend(_apply_boolean_group(df, group))

    # Deduplicate pairs (same idx_a, idx_b) — keep the highest rule_type precedence
    _TYPE_RANK = {"automatic": 0, "suspect": 1, "demoted": 2}
    seen: dict[tuple, MatchPair] = {}
    for p in all_pairs:
        key = (min(p.idx_a, p.idx_b), max(p.idx_a, p.idx_b))
        if key not in seen or _TYPE_RANK.get(p.rule_type, 9) < _TYPE_RANK.get(seen[key].rule_type, 9):
            seen[key] = p
    all_pairs = list(seen.values())

    return _demote_by_negative_rules(all_pairs, df, groups)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Build entity clusters (Union-Find)
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_clusters(pairs: list[MatchPair], n_records: int) -> dict[int, list[int]]:
    """
    Stage 2: Transitive closure via Union-Find.
    Returns {cluster_root: [record_indices]}.
    """
    uf = _UnionFind(n_records)
    for p in pairs:
        uf.union(p.idx_a, p.idx_b)
    return uf.clusters()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Create golden records (survivorship)
# ─────────────────────────────────────────────────────────────────────────────

def _survivorship(
    df: pd.DataFrame,
    record_idxs: list[int],
    source_col: Optional[str],
    source_priorities: list[str],
) -> list[GoldenField]:
    """
    For each field, pick the best value using:
      1. Highest-priority source (if source_col + priorities are provided)
      2. Most common non-null value across cluster records
      3. First non-null value (tiebreak)

    Returns list of GoldenField with full provenance.
    """
    priority_map: dict[str, int] = {}
    if source_col and source_col in df.columns and source_priorities:
        for rank, src in enumerate(source_priorities):
            priority_map[src.strip().lower()] = rank

    data_cols = [c for c in df.columns if not c.startswith("__")]
    fields: list[GoldenField] = []

    for col in data_cols:
        candidates: list[tuple] = []  # (priority, order, orig_idx, val, src_name)
        for order, orig_i in enumerate(record_idxs):
            raw = df.iloc[orig_i].get(col, "")
            val = str(raw).strip() if not pd.isna(raw) else ""
            if not val or val.lower() in ("nan", "none", "null"):
                continue
            src_name = ""
            if source_col and source_col in df.columns:
                src_name = str(df.iloc[orig_i].get(source_col, "")).strip()
            priority = priority_map.get(src_name.lower(), 999)
            candidates.append((priority, order, orig_i, val, src_name))

        if not candidates:
            continue

        if priority_map:
            # Sort by source priority first, then by original order
            candidates.sort(key=lambda x: (x[0], x[1]))
            best   = candidates[0]
            reason = "highest_priority_source" if best[0] < 999 else "only_value"
        else:
            # No priority config → most common value
            val_counts = Counter(c[3] for c in candidates)
            best_val   = val_counts.most_common(1)[0][0]
            best       = next(c for c in candidates if c[3] == best_val)
            reason     = "most_common" if len(val_counts) > 1 else "only_value"

        fields.append(GoldenField(
            field_name  = col,
            value       = best[3],
            source_idx  = best[2],
            source_name = best[4] if best[4] else f"row {best[2]}",
            reason      = reason,
        ))
    return fields


def create_golden_record(
    df: pd.DataFrame,
    record_idxs: list[int],
    source_col: Optional[str],
    source_priorities: list[str],
    all_pairs: list[MatchPair],
) -> GoldenRecord:
    """Stage 3: Create one golden record for a cluster of record indices."""
    surv_fields = _survivorship(df, record_idxs, source_col, source_priorities)

    # Determine match type for this entity
    cluster_set    = set(record_idxs)
    entity_pairs   = [p for p in all_pairs
                      if p.idx_a in cluster_set or p.idx_b in cluster_set]
    has_auto       = any(p.rule_type == "automatic" for p in entity_pairs)
    match_type     = "automatic" if has_auto else "suspect"
    matched_by     = list({p.rule_label for p in entity_pairs})

    entity_id = f"E-{min(record_idxs):06d}"
    return GoldenRecord(
        entity_id    = entity_id,
        cluster_size = len(record_idxs),
        source_idxs  = sorted(record_idxs),
        fields       = surv_fields,
        match_type   = match_type,
        matched_by   = matched_by,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    df: pd.DataFrame,
    match_rules: dict,
    source_col:          Optional[str]       = None,
    source_priorities:   Optional[list[str]] = None,
    max_golden_records:  int                 = 500,
) -> SimulationResult:
    """
    Run the full MDM simulation pipeline:

      Stage 1 — Apply match rules → MatchPairs
      Stage 2 — Union-Find clustering → entity clusters
      Stage 3 — Survivorship → golden records
      Stage 4 — Pack results into SimulationResult

    Args:
        df:                  Source dataframe
        match_rules:         Reltio matchGroups JSON dict
        source_col:          Column identifying the source system (optional)
        source_priorities:   Ordered source names, first = highest priority (optional)
        max_golden_records:  Cap on golden record creation (performance guard)

    Returns:
        SimulationResult
    """
    n            = len(df)
    src_prios    = source_priorities or []

    # Stage 1
    pairs = apply_match_rules(df, match_rules)

    # Stage 2
    clusters = build_entity_clusters(pairs, n)
    merged   = {r: idxs for r, idxs in clusters.items() if len(idxs) > 1}
    singles  = {r: idxs for r, idxs in clusters.items() if len(idxs) == 1}

    # Stage 3 — golden records for merged clusters only
    golden: list[GoldenRecord] = []
    for root, idxs in list(merged.items())[:max_golden_records]:
        gr = create_golden_record(df, idxs, source_col, src_prios, pairs)
        golden.append(gr)

    golden.sort(key=lambda x: x.cluster_size, reverse=True)

    # Stats
    n_auto    = sum(1 for gr in golden if gr.match_type == "automatic")
    n_suspect = len(golden) - n_auto
    n_single  = len(singles)
    total_ent = n_auto + n_suspect + n_single
    dedup_pct = round((n - total_ent) / n * 100, 1) if n > 0 else 0.0

    # Per-rule stats
    rule_stats: dict[str, RuleStats] = {}
    for p in pairs:
        if p.rule_uri not in rule_stats:
            rule_stats[p.rule_uri] = RuleStats(
                label=p.rule_label, type=p.rule_type, pairs=0, profiles=0
            )
        rule_stats[p.rule_uri].pairs += 1

    # Profile counts (unique records involved per rule)
    per_rule_profiles: dict[str, set] = defaultdict(set)
    for p in pairs:
        per_rule_profiles[p.rule_uri].add(p.idx_a)
        per_rule_profiles[p.rule_uri].add(p.idx_b)
    for uri, rs in rule_stats.items():
        rs.profiles = len(per_rule_profiles[uri])

    return SimulationResult(
        total_input_records  = n,
        total_entities       = total_ent,
        auto_merged_entities = n_auto,
        suspect_entities     = n_suspect,
        singleton_entities   = n_single,
        dedup_rate_pct       = dedup_pct,
        golden_records       = golden,
        match_pairs          = pairs,
        cluster_map          = merged,
        rule_stats           = rule_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers for Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def golden_records_to_df(golden: list[GoldenRecord]) -> pd.DataFrame:
    """
    Convert golden records to a flat DataFrame for Streamlit display.
    One row per entity, columns are field names + meta columns.
    """
    rows = []
    for gr in golden:
        row: dict = {
            "__entity_id":    gr.entity_id,
            "__cluster_size": gr.cluster_size,
            "__match_type":   gr.match_type,
            "__matched_by":   ", ".join(gr.matched_by),
        }
        for gf in gr.fields:
            row[gf.field_name] = gf.value
        rows.append(row)
    return pd.DataFrame(rows)


def entity_360(
    df: pd.DataFrame,
    gr: GoldenRecord,
    all_pairs: list[MatchPair],
) -> dict:
    """
    Build the Entity 360 view for a single golden record.
    Returns a dict suitable for rendering in a Streamlit expander.
    """
    source_records = [df.iloc[i].to_dict() for i in gr.source_idxs]
    cluster_pairs  = [
        p for p in all_pairs
        if p.idx_a in set(gr.source_idxs) or p.idx_b in set(gr.source_idxs)
    ]

    # Survivorship table: field → {golden_value, source, reason, conflicts}
    surv_table = []
    field_vals_by_name: dict[str, list] = defaultdict(list)
    for rec in source_records:
        for k, v in rec.items():
            if not k.startswith("__") and str(v).strip() not in ("", "nan", "None"):
                field_vals_by_name[k].append(str(v).strip())

    for gf in gr.fields:
        all_vals  = field_vals_by_name.get(gf.field_name, [])
        unique_v  = list(dict.fromkeys(all_vals))  # preserve order, deduplicate
        conflict  = len(unique_v) > 1
        surv_table.append({
            "field":         gf.field_name,
            "golden_value":  gf.value,
            "source":        gf.source_name,
            "reason":        gf.reason,
            "all_values":    unique_v,
            "conflict":      conflict,
        })

    return {
        "entity_id":      gr.entity_id,
        "cluster_size":   gr.cluster_size,
        "match_type":     gr.match_type,
        "matched_by":     gr.matched_by,
        "source_records": source_records,
        "survivorship":   surv_table,
        "pairs":          cluster_pairs,
    }
