"""
LLM utility (v3.1) — two-pass pipeline, temperature=0, full Reltio docs baked in.

v3.1 fixes:
  • max_tokens raised to 8096 for semantic analysis (truncation was the root cause
    of "Expecting property name enclosed in double quotes" errors)
  • _parse_json_robust() — 3-attempt JSON repair pipeline:
      1. Clean common defects (trailing commas, Python literals, unescaped newlines)
      2. Extract outermost { } block and retry
      3. Ask the LLM to repair the JSON (one API retry)
  • Semantic prompt now explicitly says "do not truncate — complete all columns"
  • Cache invalidated (PROMPT_VERSION bumped to v3.1)
"""

import hashlib
import json
import os
import re

import anthropic
from utils.entity_config import get_inference_guide, get_negative_rule_guidance
from __future__ import annotations


_CACHE_FILE    = ".match_rule_cache.json"
PROMPT_VERSION = "v3.5"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ── Robust JSON parsing ───────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences."""
    raw = raw.strip()
    if "```" in raw:
        # Take the block between the first and last fence pair
        parts = raw.split("```")
        # parts[1] is the content after the opening fence
        if len(parts) >= 3:
            raw = parts[1]
        elif len(parts) == 2:
            raw = parts[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    return raw.strip()


def _clean_json_text(text: str) -> str:
    """
    Fix the most common defects LLMs introduce in JSON output:
      1. Trailing commas before } or ]
      2. Python-style True / False / None
      3. Unescaped literal newlines inside string values
      4. Truncated output — close any unbalanced braces / brackets
      5. Smart/curly quotes
    """
    text = _strip_fences(text)

    # Smart quotes → straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Python literals
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b',  'null',  text)

    # Trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Unescaped literal newlines / tabs inside quoted strings
    def _escape_inner(m: re.Match) -> str:
        s = m.group(0)
        # Only touch the content between the outer quotes
        inner = s[1:-1]
        inner = inner.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return '"' + inner + '"'

    # Match JSON strings carefully (handles escaped quotes inside)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', _escape_inner, text, flags=re.DOTALL)

    # Close unbalanced braces/brackets (handles truncation)
    open_b = text.count('{') - text.count('}')
    open_k = text.count('[') - text.count(']')
    if open_b > 0 or open_k > 0:
        text = text.rstrip().rstrip(',')
        text += ']' * max(open_k, 0)
        text += '}' * max(open_b, 0)

    return text


def _parse_json_robust(raw: str, api_key: str, context: str = "") -> dict:
    """
    Parse JSON with 3 fallback attempts:
      1. Clean + direct parse
      2. Extract outermost { ... } block, clean + parse
      3. Ask the LLM to repair and parse the result

    Raises ValueError if all three fail.
    """
    # Attempt 1
    cleaned = _clean_json_text(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2 — find outermost { ... }
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(_clean_json_text(m.group(0)))
        except json.JSONDecodeError:
            pass

    # Attempt 3 — LLM repair
    client = anthropic.Anthropic(api_key=api_key)
    repair_resp = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 8096,
        temperature= 0,
        messages   = [{
            "role":    "user",
            "content": (
                "The following text is almost-valid JSON but contains a syntax error. "
                "Fix ONLY the syntax. Return ONLY the corrected raw JSON. "
                "No explanation, no markdown fences, no prose.\n\n"
                f"Context: {context}\n\n"
                f"Broken JSON (first 8000 chars):\n{raw[:8000]}"
            ),
        }],
    )
    repaired = _clean_json_text(repair_resp.content[0].text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parsing failed after 3 attempts.\n"
            f"Last error: {e}\n"
            f"Cleaned snippet (first 400 chars):\n{repaired[:400]}"
        ) from e


# ═════════════════════════════════════════════════════════════════════════════
# PASS 1 — SEMANTIC ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

_SEMANTIC_SYSTEM = """
You are a senior Master Data Management (MDM) data architect specialising in healthcare,
life sciences, and commercial data.

Your task: examine each column in the profiling summary (which includes up to 30 real
sample values from the FULL dataset) and classify it precisely so that Reltio match
rules can be generated correctly afterwards.

CRITICAL OUTPUT RULES:
  1. You MUST classify EVERY column listed. Do not stop early.
  2. Output ONLY a raw JSON object. No markdown, no code fences, no prose.
  3. Every string value in the JSON must be on a single line — no literal newlines inside strings.
  4. Do not add trailing commas.

IMPORTANT — read actual sample_values carefully. Do NOT rely on column names alone.
Many columns are misnamed or abbreviated.

{entity_inference_guide}

For EVERY column output a JSON entry with these exact fields:
  "semantic_type"       — one of the types listed below
  "reasoning"           — one sentence citing column name AND a sample value (keep it short)
  "can_be_primary"      — true or false
  "match_role"          — one of the roles listed below
  "composite_with"      — array of column names ([] if none)
  "duplicate_behaviour" — very short phrase (under 10 words)

SEMANTIC TYPES:
  strong_identifier   NPI, SSN, DEA#, EIN, MRN, UPIN — globally unique, one person.
  scoped_identifier   State license, facility code — unique only within a scope.
  person_name         FirstName, LastName, MiddleName, Prefix, Suffix.
  organization_name   Company, hospital, practice name.
  address_component   Street, city, state, ZIP, country.
  contact_info        Email, phone, fax — weak identifier, never primary.
  demographic         DOB, age, gender, race — never primary.
  taxonomy_code       Specialty, NUCC code, ICD — never primary.
  credential          MD, DO, RN, degree — never primary.
  system_field        Internal IDs, timestamps, row numbers — not matchable.
  free_text           Notes, descriptions — not matchable.
  unknown             Cannot determine.

MATCH ROLES:
  primary_strong      strong_identifier → own automatic group
  primary_scoped      scoped_identifier → own suspect group with scope constraint
  primary_name        person_name / org_name → anchor of fuzzy name group
  primary_composite   address_component → composite address group only
  supporting_exact    tightens other rules via exact constraint
  supporting_equals   used as equals/notEquals filter
  not_matchable       exclude from all rules

Output format (no extra keys, no trailing commas):
{
  "columns": {
    "ColumnName": {
      "semantic_type": "...",
      "reasoning": "...",
      "can_be_primary": true,
      "match_role": "...",
      "composite_with": [],
      "duplicate_behaviour": "..."
    }
  }
}
""".strip()

_SEMANTIC_USER = """
Entity type: {entity_type}

Read sample_values carefully for EVERY column. Classify ALL {n_cols} columns — do not stop early.

Column profiling:
{profiling_json}

Output the complete semantic analysis JSON for all {n_cols} columns now.
""".strip()



def _build_semantic_system(entity_type: str) -> str:
    """Build the semantic system prompt with entity-specific field recognition guide."""
    try:
        guide = get_inference_guide(entity_type)
    except Exception:
        guide = ""
    return _SEMANTIC_SYSTEM.replace("{entity_inference_guide}", guide)


def analyze_semantics(profiling_summary: dict, entity_type: str, api_key: str) -> dict:
    """
    Pass 1: Classify every column by reading 30 real sample values.
    Cached by SHA-256 of inputs.
    """
    cache = _load_cache()
    key   = _sha({"p": profiling_summary, "e": entity_type,
                   "step": "sem", "pv": PROMPT_VERSION})
    if key in cache:
        return cache[key]

    n_cols = len(profiling_summary)
    client = anthropic.Anthropic(api_key=api_key)
    resp   = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 8096,        # ← was 4000; raised to prevent truncation
        temperature= 0,
        system     = _build_semantic_system(entity_type),
        messages   = [{"role": "user", "content": _SEMANTIC_USER.format(
            entity_type   = entity_type,
            n_cols        = n_cols,
            profiling_json= json.dumps(profiling_summary, indent=2),
        )}],
    )

    parsed = _parse_json_robust(
        resp.content[0].text,
        api_key  = api_key,
        context  = "semantic analysis of dataset columns",
    )
    if "columns" not in parsed:
        raise ValueError("Semantic analysis missing 'columns' key")

    # Warn if the LLM skipped columns (soft check — don't hard-fail)
    classified = set(parsed["columns"].keys())
    expected   = set(profiling_summary.keys())
    missing    = expected - classified
    if missing:
        # Fill gaps with unknown so downstream never breaks
        for col in missing:
            parsed["columns"][col] = {
                "semantic_type":      "unknown",
                "reasoning":          "Not classified by LLM — added as fallback.",
                "can_be_primary":     False,
                "match_role":         "not_matchable",
                "composite_with":     [],
                "duplicate_behaviour": "unknown",
            }

    cache[key] = parsed
    _save_cache(cache)
    return parsed


# ═════════════════════════════════════════════════════════════════════════════
# PASS 2 — RULE GENERATION
# Full Reltio documentation knowledge embedded in the prompt.
# ═════════════════════════════════════════════════════════════════════════════

_RULE_SYSTEM = """
You are a Reltio MDM expert. Generate a production-quality matchGroups JSON
configuration based on the semantic analysis provided.

CRITICAL OUTPUT RULES:
  1. Output ONLY raw JSON. No markdown, no code fences, no prose.
  2. Every string value must be on one line — no literal newlines inside strings.
  3. No trailing commas.
  4. Complete the full JSON — do not truncate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELTIO MATCHING — REFERENCE (from official docs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MATCH GROUP TYPES
  automatic       Boolean. TRUE → merge. Only for strong identifiers (near-zero false-positive risk).
  suspect         Boolean. TRUE → queue for review by data steward.
  relevance_based Arithmetic score 0-1. Score range determines merge vs queue-for-review.
                  Use when weighting multiple soft attributes. Replaces needing separate auto+suspect.
  negativeRule    If TRUE → demotes ALL merge directives from ALL other rules to queue-for-review.
                  Use to PREVENT merging when records clearly differ on a key field.
                  No "type" field — uses "negativeRule" key instead of "rule".

DIRECTIVE PRECEDENCE
  merge > queue_for_review, EXCEPT if negativeRule fires → all merges become queue_for_review.

SCOPE (default ALL)   ALL | INTERNAL | EXTERNAL | NONE
useOvOnly             Set "useOvOnly": "true" on every group (strongly recommended by Reltio).

TOKEN CLASSES
  com.reltio.match.token.ExactMatchToken           — identifiers and exact fields
  com.reltio.match.token.DoubleMetaphoneMatchToken — name fields with phonetic variation

COMPARATOR CLASSES (boolean)
  com.reltio.match.comparator.ExactComparator              — exact string match
  com.reltio.match.comparator.DoubleMetaphoneComparator    — phonetic name comparison

COMPARATOR CLASSES (relevance_based — return 0.0 to 1.0)
  com.reltio.match.comparator.JaroComparator          — short string similarity
  com.reltio.match.comparator.LevenshteinComparator   — edit-distance similarity
  com.reltio.match.comparator.BasicStringComparator   — 1 if identical, 0 otherwise

FORMULA OPERANDS (boolean rules)
  and, or, exact, fuzzy, exactOrNull, exactOrAllNull, notExact,
  equals, notEquals, nullValues, multi

NEGATIVE RULE OPERANDS (only these):
  notExactSame, notFuzzySame, or, and, equals, notEquals, nullValues
  Does NOT support: exact, fuzzy, not, multi

scoreStandalone / scoreIncremental — always 0 unless implementing composite scoring.

RELEVANCE-BASED STRUCTURE
  "weights": [{"attribute": "<uri>", "weight": 0.0-1.0}, ...]
  "actionThresholds": [
    {"type": "queue_for_review", "threshold": "0.0-0.6"},
    {"type": "auto_merge",       "threshold": "0.6-1.0"}
  ]
  Thresholds must span 0.0-1.0 with no gaps and touching endpoints only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[R1] strong_identifier → type=automatic, ExactMatchToken + ExactComparator
     "and": {"exact": ["<id_uri>"]}

[R2] scoped_identifier → type=suspect, ExactMatchToken + ExactComparator
     "and": {"exact": ["<id_uri>", "<scope_uri>"]}
     scope_uri = the state/country column. If missing, still create rule.

[R3] person_name → ONE combined suspect group only. Never create two name rules.
     Use DOB and MiddleName as "exactOrNull" (not "exact") so missing values don't
     block legitimate matches. This is the single correct structure:
       matchTokenClasses: DoubleMetaphoneMatchToken on LastName
       comparatorClasses: DoubleMetaphoneComparator on FirstName
       "and": {
         "exact":       ["<LastName_uri>"],
         "fuzzy":       ["<FirstName_uri>"],
         "exactOrNull": ["<DOB_uri>", "<MiddleName_uri>"]   ← only if those columns exist
       }
     ❌ DO NOT create a second name rule "without DOB". One name rule only.
     ❌ DO NOT create a relevance_based rule that duplicates name + DOB matching.

[R4] organization_name → type=suspect, DoubleMetaphoneMatchToken + DoubleMetaphoneComparator
     "and": {"fuzzy": ["<OrgName_uri>"]}

[R5] address_component → ONLY if street + city + state/ZIP all present
     type=suspect, "and": {"exact": ["<Street>", "<City>", "<State>"]}

[R6] relevance_based → DO NOT generate unless there are 4+ soft attributes with NO
     strong_identifier or person_name rules possible. In practice, if you have already
     generated rules under R1-R5, do NOT add a relevance_based rule — it would
     duplicate coverage and create false positives.
     If you do generate one, ALL of weights/comparatorClasses/matchTokenClasses/
     actionThresholds MUST be nested INSIDE the "rule" key — never top-level on the group.

[R7] negativeRule — STRICT eligibility. Only create a negativeRule for a field when:
     (a) The field is reliably populated (p_missing < 0.3) across your dataset, AND
     (b) The field has stable, standardised values (not free text, not multi-system codes), AND
     (c) A difference in that field definitively means two records are different people.

     {entity_negative_rule_guidance}

     REQUIRED negativeRule structure:
       "uri":   "configuration/entityTypes/{EntityType}/matchGroups/NegativeRuleOn{FieldName}"
       "label": "Negative Rule - Do not merge if {FieldName} differs"
       "scope": "ALL"
       "negativeRule": { "and": { "notExactSame": ["<field_uri>"] } }
     Do NOT add "type", "rule", "scoreStandalone", or "scoreIncremental" to a negativeRule group.

[R8] NEVER create a standalone group for:
     contact_info, demographic, taxonomy_code, credential,
     single address_component, system_field, free_text.

[R9] Every NON-negativeRule group MUST have ALL of these fields — no exceptions:
     "uri":              "configuration/entityTypes/..."  (unique, descriptive)
     "label":            "Human-readable description of what this rule matches"
     "type":             "automatic" | "suspect" | "relevance_based"
     "scope":            "ALL"
     "useOvOnly":        "true"
     "scoreStandalone":  0
     "scoreIncremental": 0

[R10] DEDUPLICATION CHECK — before finalising, verify:
     (a) No two rules match on the same primary field combination.
         If you have "name + DOB" in one rule, do NOT add "name without DOB" as another.
     (b) No relevance_based rule covers attributes already matched by a boolean rule.
     (c) Total match groups should typically be 4–8 for a well-modelled entity.
         If you have generated more than 8, review for duplicates and remove them.

URI PATTERNS:
  Group:     configuration/entityTypes/{EntityType}/matchGroups/MatchOn{DescriptiveName}
  Attribute: configuration/entityTypes/{EntityType}/attributes/{ColumnName}
""".strip()

_RULE_USER = """
Entity type: {entity_type}

Semantic analysis:
{semantic_json}

Profiling summary (verify with sample_values):
{profiling_json}

Generate the complete matchGroups JSON now.
""".strip()


def _build_rule_system(entity_type: str) -> str:
    """Build rule system prompt with entity-specific negative rule guidance."""
    guidance = get_negative_rule_guidance(entity_type)
    return _RULE_SYSTEM.replace("{entity_negative_rule_guidance}", guidance)


def generate_match_rules(
    profiling_summary: dict,
    semantic_analysis: dict,
    entity_type: str,
    api_key: str,
) -> dict:
    """
    Pass 2: Generate matchGroups JSON from semantic analysis.
    Cached by SHA-256 of inputs.
    """
    cache = _load_cache()
    key   = _sha({"s": semantic_analysis, "e": entity_type,
                   "step": "rules", "pv": PROMPT_VERSION})
    if key in cache:
        return cache[key]

    client = anthropic.Anthropic(api_key=api_key)
    resp   = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 8096,
        temperature= 0,
        system     = _build_rule_system(entity_type),
        messages   = [{"role": "user", "content": _RULE_USER.format(
            entity_type   = entity_type,
            semantic_json = json.dumps(semantic_analysis, indent=2),
            profiling_json= json.dumps(profiling_summary, indent=2),
        )}],
    )

    parsed = _parse_json_robust(
        resp.content[0].text,
        api_key = api_key,
        context = "Reltio matchGroups JSON generation",
    )

    if "matchGroups" not in parsed:
        raise ValueError("Response missing 'matchGroups' key")
    if not isinstance(parsed["matchGroups"], list):
        raise ValueError("'matchGroups' must be an array")

    for g in parsed["matchGroups"]:
        is_neg = "negativeRule" in g

        # Auto-repair: derive label from uri if missing, or vice-versa
        if "uri" not in g and "label" in g:
            slug = re.sub(r"[^A-Za-z0-9]", "", g["label"].title().replace(" ", ""))
            et   = entity_type.replace(" ", "")
            prefix = "NegativeRuleOn" if is_neg else "MatchOn"
            g["uri"] = f"configuration/entityTypes/{et}/matchGroups/{prefix}{slug}"

        if "label" not in g and "uri" in g:
            # Derive a readable label from the last path segment of the uri
            slug  = g["uri"].rstrip("/").split("/")[-1]
            # Insert spaces before capital letters: "MatchOnNPI" → "Match On N P I"
            # Better: just strip MatchOn/NegativeRuleOn prefix
            for pfx in ("NegativeRuleOn", "MatchOn"):
                if slug.startswith(pfx):
                    slug = slug[len(pfx):]
                    break
            readable = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", slug)
            g["label"] = ("Negative Rule - " if is_neg else "Match on ") + readable

        # Hard-fail only if BOTH uri and label are still missing after repair
        for req in ("uri", "label"):
            if req not in g:
                raise ValueError(
                    f"Match group is missing both 'uri' and 'label' — "
                    f"cannot auto-repair. Group keys present: {list(g.keys())}"
                )

        # ── Auto-repair: relevance_based keys placed at top level ──────────
        # The LLM sometimes puts weights/comparatorClasses/matchTokenClasses/
        # actionThresholds directly on the group instead of inside "rule".
        # Detect and fix this before validation.
        _RULE_LEVEL_KEYS = {
            "weights", "comparators", "comparatorClasses",
            "matchTokenClasses", "matchTokens", "actionThresholds", "and", "or",
            "exact", "fuzzy", "ignoreInToken", "cleanse",
        }
        if not is_neg and "rule" not in g:
            misplaced = {k: v for k, v in g.items() if k in _RULE_LEVEL_KEYS}
            if misplaced:
                # Move misplaced keys into a new "rule" dict
                g["rule"] = misplaced
                for k in misplaced:
                    del g[k]
                # Normalise key names the LLM sometimes uses
                rule = g["rule"]
                if "comparators" in rule and "comparatorClasses" not in rule:
                    rule["comparatorClasses"] = rule.pop("comparators")
                if "matchTokens" in rule and "matchTokenClasses" not in rule:
                    rule["matchTokenClasses"] = rule.pop("matchTokens")

        # Must have either rule or negativeRule after repair attempt
        if not is_neg and "rule" not in g:
            raise ValueError(
                f"Match group '{g.get('uri')}' has no 'rule' and no repairable keys. "
                f"Keys present: {list(g.keys())}"
            )

        # Auto-fill common missing scalar fields with safe defaults
        if not is_neg:
            g.setdefault("scope",            "ALL")
            g.setdefault("useOvOnly",        "true")
            g.setdefault("scoreStandalone",  0)
            g.setdefault("scoreIncremental", 0)
        else:
            g.setdefault("scope", "ALL")

    cache[key] = parsed
    _save_cache(cache)
    return parsed

_EVIDENCE_SYSTEM = """
You are a Reltio MDM expert generating production-quality matchGroups JSON.

You have TWO inputs:
  1. Semantic analysis  — LLM-classified column types and match roles
  2. Match evidence     — empirically extracted from REAL confirmed duplicate pairs
                          found in the actual dataset by vector similarity + Claude adjudication

EVIDENCE FIELDS:
  frequency        — fraction of confirmed MATCH pairs where this field agreed (0.0–1.0)
  match_type       — "exact" | "fuzzy_phonetic" | "mixed"
  weight_suggestion — suggested relevance weight (0.0–1.0)
  examples         — real value pairs that triggered the match

EVIDENCE INTERPRETATION RULES:
  • frequency ≥ 0.85  → HIGH-VALUE field. Use as primary match criterion.
  • frequency 0.50–0.84 → SUPPORTING field. Include in composite rules or exactOrNull.
  • frequency 0.25–0.49 → WEAK field. Use only as exactOrNull or ignore.
  • frequency < 0.25    → NOISE. Do not use in rules.

  • match_type = "exact"        → use ExactMatchToken + ExactComparator
  • match_type = "fuzzy_phonetic" → use DoubleMetaphoneMatchToken + DoubleMetaphoneComparator
  • match_type = "mixed"        → prefer fuzzy token with exact fallback

PRIORITY: Evidence OVERRIDES schema-inferred match roles for fields with frequency ≥ 0.5.
  Example: if schema says "specialty = taxonomy_code (not_matchable)"
           but evidence says "specialty frequency = 0.78, match_type = exact"
           → include specialty as a supporting exact field in the name+DOB rule.

Follow all standard Reltio rules (R1–R10):
  [R1] strong_identifier → type=automatic, ExactMatchToken
  [R2] scoped_identifier → type=suspect with scope constraint
  [R3] person_name → ONE combined suspect group (LastName exact + FirstName fuzzy + DOB/Middle exactOrNull)
  [R4] organization_name → suspect, fuzzy
  [R5] address → only if street+city+state present
  [R6] relevance_based → only if 4+ soft attributes with NO strong/name rules possible
  [R7] negativeRule → only eligible fields (see entity guidance below)
  [R8] NEVER standalone group for: contact_info, demographic, taxonomy_code, system_field
  [R9] Non-negative groups MUST have: uri, label, type, scope=ALL, useOvOnly=true,
       scoreStandalone=0, scoreIncremental=0
  [R10] 4–8 match groups total; no duplicates

CRITICAL OUTPUT RULES:
  • Output ONLY raw JSON. No markdown, no fences, no prose.
  • Every string on one line — no literal newlines inside strings.
  • No trailing commas.
  • Complete the full JSON — do not truncate.
""".strip()

_EVIDENCE_USER = """
Entity type: {entity_type}

── Semantic analysis (column classifications) ───────────────────────────────
{semantic_json}

── Match evidence (empirical, from confirmed duplicate pairs in actual data) ─
Confirmed MATCH pairs analysed: {n_match}
Field-level agreement evidence (sorted by frequency):
{evidence_json}

── Profiling summary (for context) ─────────────────────────────────────────
{profiling_json}

Generate the complete matchGroups JSON now.
Use the evidence frequency and match_type to:
  1. Confirm or correct schema-inferred roles
  2. Set exact vs fuzzy matching per field
  3. Choose primary vs supporting fields based on frequency
  4. Skip fields with frequency < 0.25 from all rules
""".strip()


def generate_evidence_driven_rules(
    profiling_summary: dict,
    semantic_analysis: dict,
    match_evidence:    dict,
    entity_type:       str,
    api_key:           str,
    n_match_pairs:     int = 0,
) -> dict:
    """
    Pass 2.5: Generate match rules grounded in actual data evidence.

    If match_evidence is empty (no confirmed duplicates found by vectorizer),
    falls back transparently to generate_match_rules() (Pass 2).

    Args:
        profiling_summary:  Output of profiler.profile_dataframe()
        semantic_analysis:  Output of analyze_semantics() (Pass 1)
        match_evidence:     Output of vectorizer.extract_match_evidence()
        entity_type:        e.g. "HCP", "Customer"
        api_key:            Anthropic API key
        n_match_pairs:      Number of MATCH pairs used to build evidence (for prompt context)

    Returns:
        matchGroups JSON dict — same schema as generate_match_rules() output.
    """
    # Fallback: no evidence → standard Pass 2
    if not match_evidence:
        return generate_match_rules(profiling_summary, semantic_analysis, entity_type, api_key)

    cache = _load_cache()
    key   = _sha({
        "s": semantic_analysis,
        "ev": match_evidence,
        "e": entity_type,
        "step": "evidence_rules",
        "pv": PROMPT_VERSION,
    })
    if key in cache:
        return cache[key]

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Build entity-specific negative rule guidance (same as Pass 2)
    from utils.entity_config import get_negative_rule_guidance
    neg_guidance = get_negative_rule_guidance(entity_type)

    system_prompt = _EVIDENCE_SYSTEM
    if neg_guidance:
        system_prompt += f"\n\nNEGATIVE RULE ELIGIBILITY for {entity_type}:\n{neg_guidance}"

    resp = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 8096,
        temperature= 0,
        system     = system_prompt,
        messages   = [{
            "role": "user",
            "content": _EVIDENCE_USER.format(
                entity_type   = entity_type,
                semantic_json = json.dumps(semantic_analysis, indent=2),
                evidence_json = json.dumps(match_evidence, indent=2),
                profiling_json= json.dumps(profiling_summary, indent=2),
                n_match       = n_match_pairs,
            ),
        }],
    )

    parsed = _parse_json_robust(
        resp.content[0].text,
        api_key  = api_key,
        context  = f"evidence-driven Reltio matchGroups for {entity_type}",
    )

    if "matchGroups" not in parsed:
        raise ValueError("Evidence-driven response missing 'matchGroups' key — "
                         "falling back to standard Pass 2")

    # Apply same auto-repair logic as generate_match_rules()
    _RULE_LEVEL_KEYS = {
        "weights", "comparators", "comparatorClasses",
        "matchTokenClasses", "matchTokens", "actionThresholds",
        "and", "or", "exact", "fuzzy", "ignoreInToken", "cleanse",
    }
    for g in parsed["matchGroups"]:
        is_neg = "negativeRule" in g

        # Auto-derive uri from label and vice versa
        if "uri" not in g and "label" in g:
            slug   = re.sub(r"[^A-Za-z0-9]", "", g["label"].title().replace(" ", ""))
            et     = entity_type.replace(" ", "")
            prefix = "NegativeRuleOn" if is_neg else "MatchOn"
            g["uri"] = f"configuration/entityTypes/{et}/matchGroups/{prefix}{slug}"

        if "label" not in g and "uri" in g:
            slug = g["uri"].rstrip("/").split("/")[-1]
            for pfx in ("NegativeRuleOn", "MatchOn"):
                if slug.startswith(pfx):
                    slug = slug[len(pfx):]
                    break
            readable = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", slug)
            g["label"] = ("Negative Rule - " if is_neg else "Match on ") + readable

        if not is_neg and "rule" not in g:
            misplaced = {k: v for k, v in g.items() if k in _RULE_LEVEL_KEYS}
            if misplaced:
                g["rule"] = misplaced
                for k in misplaced:
                    del g[k]
                rule = g["rule"]
                if "comparators" in rule and "comparatorClasses" not in rule:
                    rule["comparatorClasses"] = rule.pop("comparators")
                if "matchTokens" in rule and "matchTokenClasses" not in rule:
                    rule["matchTokenClasses"] = rule.pop("matchTokens")

        if not is_neg:
            g.setdefault("scope",            "ALL")
            g.setdefault("useOvOnly",        "true")
            g.setdefault("scoreStandalone",  0)
            g.setdefault("scoreIncremental", 0)
        else:
            g.setdefault("scope", "ALL")

    cache[key] = parsed
    _save_cache(cache)
    return parsed


# ── Helper: evidence summary for UI display ───────────────────────────────────

def format_evidence_summary(match_evidence: dict) -> list[dict]:
    """
    Convert match_evidence dict to a list of dicts for Streamlit display.
    Returns rows sorted by frequency descending.
    """
    rows = []
    for field, ev in match_evidence.items():
        rows.append({
            "Field":             field,
            "Frequency":         f"{ev['frequency']:.0%}",
            "Match type":        ev["match_type"].replace("_", " "),
            "Exact / Fuzzy":     f"{ev['exact_matches']} / {ev['fuzzy_matches']}",
            "Compared pairs":    ev["total_compared"],
            "Weight suggestion": ev["weight_suggestion"],
            "Strength":          (
                "🟢 High"   if ev["frequency"] >= 0.85 else
                "🟡 Medium" if ev["frequency"] >= 0.50 else
                "🟠 Weak"
            ),
        })
    return rows

