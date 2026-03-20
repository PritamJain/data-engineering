"""
Pre-MDM Sandbox — Reltio & Semarchy Match Rule Generator
6-step wizard:
  01 Ingest       file upload
  02 Profile      full-dataset stats + issue flags
  03 Data Quality AI-suggested DQ fixes
  04 Match Rules  LLM semantic analysis + rule generation + NL refinement
  05 Simulate     match count simulation against real data
  06 Export       Reltio matchGroups JSON + Semarchy YAML + profiling CSV
"""

import io
import json
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.llm import analyze_semantics, generate_match_rules
from utils.profiler import profile_dataframe, simulate_match_counts
from utils.semarchy import reltio_to_semarchy_yaml

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pre-MDM Sandbox",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.stApp{background:#0e0f14;color:#e2e8f0;}
[data-testid="stSidebar"]{background:#111318;border-right:1px solid #1e2130;}

.step-badge{display:inline-block;background:#1e2130;color:#6ee7f7;
  font-family:'IBM Plex Mono',monospace;font-size:.7rem;padding:2px 10px;
  border-radius:20px;margin-bottom:.5rem;letter-spacing:.08em;}
.section-title{font-size:1.4rem;font-weight:700;letter-spacing:-.02em;
  color:#e2e8f0;margin-bottom:.15rem;}
.section-sub{font-size:.82rem;color:#64748b;font-family:'IBM Plex Mono',monospace;
  margin-bottom:1.2rem;}

[data-testid="stMetric"]{background:#161821;border:1px solid #1e2130;
  border-radius:10px;padding:12px 16px;}
[data-testid="stMetricLabel"]{font-size:.7rem!important;color:#64748b!important;
  font-family:'IBM Plex Mono',monospace;}
[data-testid="stMetricValue"]{font-size:1.5rem!important;font-weight:700!important;
  color:#6ee7f7!important;}

.pill{display:inline-block;padding:3px 10px;border-radius:4px;
  font-size:.75rem;font-family:'IBM Plex Mono',monospace;}
.pill-green {background:#052e16;color:#4ade80;}
.pill-amber {background:#451a03;color:#fb923c;}
.pill-red   {background:#3b0a0a;color:#f87171;}
.pill-blue  {background:#0c1a2e;color:#38bdf8;}
.pill-purple{background:#2d1b69;color:#a78bfa;}

.rule-card{background:#161821;border:1px solid #1e2130;
  border-left:3px solid #a78bfa;border-radius:0 10px 10px 0;
  padding:1rem 1.2rem .8rem;margin-bottom:.9rem;}
.rule-card.neg{border-left-color:#f87171;}
.rule-card.auto-rule{border-left-color:#4ade80;}
.rule-title{font-weight:600;font-size:.9rem;color:#e2e8f0;margin-bottom:.3rem;}
.rule-meta{font-size:.75rem;color:#64748b;}
.match-stats{display:flex;gap:1.2rem;margin-top:.6rem;padding-top:.5rem;
  border-top:1px solid #1e2130;}
.ms-item{display:flex;flex-direction:column;}
.ms-val{font-size:1rem;font-weight:700;color:#6ee7f7;
  font-family:'IBM Plex Mono',monospace;}
.ms-val.warn{color:#fb923c;}
.ms-lbl{font-size:.65rem;color:#475569;margin-top:1px;}

.summary-bar{display:flex;gap:1.5rem;flex-wrap:wrap;background:#161821;
  border:1px solid #1e2130;border-radius:10px;
  padding:.9rem 1.4rem;margin-bottom:1.2rem;}
.sb-item{display:flex;flex-direction:column;}
.sb-val{font-size:1.2rem;font-weight:700;font-family:'IBM Plex Mono',monospace;
  color:#a78bfa;}
.sb-lbl{font-size:.7rem;color:#475569;}

.nl-you{background:#1e2130;border-radius:8px;padding:.5rem .9rem;
  font-size:.82rem;margin-bottom:.2rem;color:#e2e8f0;}
.nl-sys{background:#0f2a1a;border-radius:8px;padding:.5rem .9rem;
  font-size:.82rem;margin-bottom:.5rem;color:#34d399;}

.sem-card{background:#161821;border:1px solid #1e2130;border-radius:10px;
  padding:.8rem 1rem;margin-bottom:.5rem;}
.sem-reason{font-size:.75rem;color:#64748b;margin-top:.25rem;line-height:1.5;}

.tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.7rem;
  font-family:'IBM Plex Mono',monospace;margin-right:3px;vertical-align:middle;}
.tag-auto   {background:#052e16;color:#4ade80;}
.tag-suspect{background:#2d1b69;color:#a78bfa;}
.tag-rel    {background:#451a03;color:#fb923c;}
.tag-neg    {background:#3b0a0a;color:#f87171;}

.stype-strong{background:#052e16;color:#4ade80;}
.stype-scoped{background:#0f2a1a;color:#34d399;}
.stype-name  {background:#2d1b69;color:#a78bfa;}
.stype-org   {background:#1e1240;color:#c4b5fd;}
.stype-addr  {background:#1c1917;color:#d4a574;}
.stype-weak  {background:#1e1a0f;color:#fbbf24;}
.stype-other {background:#1e2130;color:#64748b;}

.export-card{background:#161821;border:1px solid #1e2130;border-radius:12px;
  padding:1.4rem;margin-bottom:1rem;}
.export-title{font-weight:600;font-size:1rem;color:#e2e8f0;margin-bottom:.3rem;}
.export-sub{font-size:.78rem;color:#64748b;margin-bottom:1rem;line-height:1.5;}
</style>
""", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
DEFAULTS = {
    "step":        1,
    "df":          None,
    "raw_profile": None,
    "dq_fixes":    None,
    "semantic":    None,
    "match_rules": None,
    "counts":      None,
    "nl_history":  [],
    "entity_type": "HCP",
    "api_key":     "",
    "null_pct":    80,
    "card_pct":    90,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

STEPS = {
    1: "01 · Ingest",
    2: "02 · Profile",
    3: "03 · Data Quality",
    4: "04 · Match Rules",
    5: "05 · Simulate",
    6: "06 · Export",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔗 Pre-MDM Sandbox")
    st.caption("Data prep · Reltio & Semarchy\nmatch rule generation via Claude")
    st.markdown("---")

    for num, label in STEPS.items():
        locked = num > st.session_state.step
        is_cur = num == st.session_state.step
        if st.button(label, key=f"nav_{num}", disabled=locked,
                     use_container_width=True,
                     type="primary" if is_cur else "secondary"):
            st.session_state.step = num
            st.rerun()

    st.markdown("---")
    st.session_state.entity_type = st.text_input(
        "Entity type", value=st.session_state.entity_type)
    try:
        st.session_state.api_key = st.secrets["ANTHROPIC_API_KEY"]
        st.success("API key loaded ✓", icon="🔑")
    except Exception:
        st.session_state.api_key = st.text_input(
            "Anthropic API key", type="password", value=st.session_state.api_key,
            help="Required for steps 04+")
    st.markdown("---")
    if st.button("🔄 Reset everything", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _tag(label, cls):
    return f'<span class="tag {cls}">{label}</span>'

def _type_tag(t):
    m = {"automatic": "tag-auto", "suspect": "tag-suspect",
         "relevance_based": "tag-rel"}
    return _tag(t, m.get(t, "tag-suspect"))

_STYPE_CSS = {
    "strong_identifier": "stype-strong", "scoped_identifier": "stype-scoped",
    "person_name": "stype-name",         "organization_name": "stype-org",
    "address_component": "stype-addr",   "contact_info": "stype-weak",
    "demographic": "stype-weak",
}

def _quick_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight profile for the UI table in step 2."""
    rows = []
    for col in df.columns:
        null_pct  = round(df[col].isna().mean() * 100, 1)
        uniq_pct  = round(df[col].nunique() / max(len(df), 1) * 100, 1)
        dtype     = str(df[col].dtype)
        issues    = []
        if null_pct > 5:
            issues.append(f"High nulls ({null_pct}%)")
        if df[col].dtype == object:
            s = df[col].dropna().astype(str)
            if s.str.lower().ne(s).any() and s.str.upper().ne(s).any():
                issues.append("Mixed casing")
        rows.append({
            "Field": col, "Type": dtype,
            "Null %": null_pct, "Unique %": uniq_pct,
            "Issues": ", ".join(issues) if issues else "✓ Clean",
            "_has_issues": len(issues) > 0,
        })
    return pd.DataFrame(rows)

def _suggest_dq(df: pd.DataFrame) -> pd.DataFrame:
    fixes = []
    for col in df.columns:
        cl = col.lower()
        if df[col].dtype != object:
            continue
        s = df[col].dropna().astype(str)
        if any(x in cl for x in ("phone", "mobile", "fax", "tel")):
            fixes.append({"Field": col, "Fix": "Normalise to E.164 format",
                          "Records": len(s), "Apply": False})
        if any(x in cl for x in ("first", "last", "name", "middle")):
            mixed = s[s.str.lower().ne(s) & s.str.upper().ne(s)]
            if len(mixed):
                fixes.append({"Field": col, "Fix": "Standardise casing (Title Case)",
                              "Records": len(mixed), "Apply": False})
        if any(x in cl for x in ("email", "mail")):
            bad = s[~s.str.contains(r"^[^@]+@[^@]+\.[^@]+$", na=False)]
            if len(bad):
                fixes.append({"Field": col, "Fix": "Flag invalid email addresses",
                              "Records": len(bad), "Apply": False})
        if any(x in cl for x in ("dob", "birth", "date")):
            fixes.append({"Field": col, "Fix": "Standardise to ISO 8601 (YYYY-MM-DD)",
                          "Records": len(s), "Apply": False})
        if any(x in cl for x in ("address", "street", "addr")):
            fixes.append({"Field": col, "Fix": "Expand abbreviations (St→Street, Ave→Avenue)",
                          "Records": int(len(s) * 0.15), "Apply": False})
    if not fixes:
        fixes.append({"Field": "—", "Fix": "No critical issues detected",
                      "Records": 0, "Apply": False})
    return pd.DataFrame(fixes)

# ── STEP 1: Ingest ────────────────────────────────────────────────────────────
if st.session_state.step == 1:
    st.markdown('<div class="step-badge">STEP 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV, JSON, or Parquet file to begin</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your file here", type=["csv", "json", "parquet"],
                                label_visibility="collapsed")
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):     df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(".json"):  df = pd.read_json(uploaded)
            else:                                  df = pd.read_parquet(uploaded)
            st.success(f"✓ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df.head(8), use_container_width=True)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.null_pct = st.slider(
            "Skip column if null% >", 0, 100, st.session_state.null_pct, 5)
    with c2:
        st.session_state.card_pct = st.slider(
            "Flag as high-cardinality if unique% >", 50, 100,
            st.session_state.card_pct, 5)

    if st.session_state.df is not None:
        if st.button("▶  Run data profiling", type="primary", use_container_width=True):
            with st.spinner("Profiling schema and statistics…"):
                df = st.session_state.df
                st.session_state.raw_profile  = _quick_profile(df)
                st.session_state.dq_fixes     = _suggest_dq(df)
                st.session_state.step         = 2
            st.rerun()
    else:
        st.info("Upload a file above to begin.")

# ── STEP 2: Profile ───────────────────────────────────────────────────────────
elif st.session_state.step == 2:
    df      = st.session_state.df
    profile = st.session_state.raw_profile

    st.markdown('<div class="step-badge">STEP 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Profiling report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{len(df):,} records · '
                f'{len(df.columns)} columns analysed</div>', unsafe_allow_html=True)

    issues_n     = int(profile["_has_issues"].sum())
    completeness = round((1 - df.isna().mean().mean()) * 100, 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records",      f"{len(df):,}")
    c2.metric("Columns",      len(df.columns))
    c3.metric("Quality issues", issues_n)
    c4.metric("Completeness", f"{completeness}%")

    st.markdown("---")
    display = profile.drop(columns=["_has_issues"]).copy()

    def _hi(row):
        base = [""] * 5
        if row["Issues"] != "✓ Clean":
            base[4] = "background-color:#451a03;color:#fb923c"
        else:
            base[4] = "background-color:#052e16;color:#4ade80"
        return base

    st.dataframe(display.style.apply(_hi, axis=1),
                 use_container_width=True, hide_index=True)

    if issues_n:
        st.markdown("**Issues detected:**")
        for _, row in profile[profile["_has_issues"]].iterrows():
            st.markdown(f"- `{row['Field']}` — {row['Issues']}")

    st.markdown("---")
    if st.button("▶  Proceed to data quality", type="primary", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

# ── STEP 3: Data Quality ──────────────────────────────────────────────────────
elif st.session_state.step == 3:
    st.markdown('<div class="step-badge">STEP 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data quality & enrichment</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI-suggested corrections before matching</div>',
                unsafe_allow_html=True)

    fixes_df = st.session_state.dq_fixes.copy()
    ca, cb   = st.columns([1, 5])
    with ca:
        if st.button("✅ Apply all"):
            fixes_df["Apply"] = True
            st.session_state.dq_fixes = fixes_df
    with cb:
        if st.button("↩ Clear all"):
            fixes_df["Apply"] = False
            st.session_state.dq_fixes = fixes_df

    st.markdown("---")
    for i, row in fixes_df.iterrows():
        c1, c2, c3, c4 = st.columns([0.5, 3.5, 2, 1])
        with c1:
            checked = st.checkbox("", value=row["Apply"], key=f"fix_{i}",
                                  label_visibility="collapsed")
            fixes_df.at[i, "Apply"] = checked
        with c2:
            st.markdown(f"~~{row['Fix']}~~" if checked else row["Fix"])
        with c3:
            st.caption(f"`{row['Field']}` · {row['Records']:,} records")
        with c4:
            cls = "pill-green" if checked else "pill-amber"
            txt = "Applied" if checked else "Pending"
            st.markdown(f'<span class="pill {cls}">{txt}</span>', unsafe_allow_html=True)

    st.session_state.dq_fixes = fixes_df
    applied = int(fixes_df["Apply"].sum())
    st.markdown("---")
    st.caption(f"{applied} of {len(fixes_df)} fixes selected")

    if st.button("▶  Configure match rules", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

# ── STEP 4: Match Rules ───────────────────────────────────────────────────────
elif st.session_state.step == 4:
    st.markdown('<div class="step-badge">STEP 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Match rule generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">LLM semantic analysis → Reltio matchGroups + Semarchy YAML</div>',
                unsafe_allow_html=True)

    entity_type   = st.session_state.entity_type
    api_key       = st.session_state.api_key
    df            = st.session_state.df

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to continue.")
        st.stop()

    run_btn = st.button("⚡  Run LLM pipeline", type="primary", use_container_width=True)

    if run_btn:
        cfg = {
            "entity_type":           entity_type,
            "null_threshold":        st.session_state.null_pct / 100,
            "cardinality_threshold": st.session_state.card_pct / 100,
        }
        with st.status("Running LLM pipeline…", expanded=True) as status:
            st.write(f"📊 Profiling {df.shape[1]} columns across {df.shape[0]:,} rows…")
            prof = profile_dataframe(df, cfg)
            skipped = df.shape[1] - len(prof)
            st.write(f"✅ {len(prof)} columns profiled"
                     + (f" ({skipped} skipped — too sparse)" if skipped else ""))

            st.write("🧠 Classifying column semantics from real data…")
            sem = analyze_semantics(prof, entity_type, api_key)
            cols = sem.get("columns", {})
            n_pri = sum(1 for c in cols.values() if c.get("can_be_primary"))
            st.write(f"✅ {len(cols)} columns classified, {n_pri} primary candidates")

            st.write("⚙️  Generating matchGroups JSON…")
            rules = generate_match_rules(prof, sem, entity_type, api_key)
            groups = rules.get("matchGroups", [])
            st.write(f"✅ {len(groups)} match group(s) generated")

            status.update(label="✅ Pipeline complete!", state="complete")

        st.session_state.match_rules = rules
        st.session_state.semantic    = sem

    # ── Show rules ────────────────────────────────────────────────────────────
    if st.session_state.match_rules:
        rules  = st.session_state.match_rules
        sem    = st.session_state.get("semantic", {})
        groups = rules.get("matchGroups", [])

        n_auto    = sum(1 for g in groups if g.get("type") == "automatic")
        n_suspect = sum(1 for g in groups if g.get("type") == "suspect")
        n_rel     = sum(1 for g in groups if g.get("type") == "relevance_based")
        n_neg     = sum(1 for g in groups if "negativeRule" in g)

        st.markdown(f"""<div class="summary-bar">
          <div class="sb-item"><span class="sb-val">{len(groups)}</span>
            <span class="sb-lbl">total groups</span></div>
          <div class="sb-item"><span class="sb-val">{n_auto}</span>
            <span class="sb-lbl">automatic</span></div>
          <div class="sb-item"><span class="sb-val">{n_suspect}</span>
            <span class="sb-lbl">suspect</span></div>
          <div class="sb-item"><span class="sb-val">{n_rel}</span>
            <span class="sb-lbl">relevance</span></div>
          <div class="sb-item"><span class="sb-val">{n_neg}</span>
            <span class="sb-lbl">negative rules</span></div>
        </div>""", unsafe_allow_html=True)

        for g in groups:
            is_neg   = "negativeRule" in g
            rtype    = g.get("type", "")
            label    = g.get("label", g.get("uri", "").split("/")[-1])
            scope    = g.get("scope", "ALL")
            use_ov   = g.get("useOvOnly", "false")
            card_cls = ("neg" if is_neg else
                        "auto-rule" if rtype == "automatic" else "")
            type_tag = (_tag("negativeRule", "tag-neg") if is_neg
                        else _type_tag(rtype))
            ov_badge = (_tag("useOvOnly ✓", "tag-suspect")
                        if use_ov == "true" else "")
            st.markdown(f"""<div class="rule-card {card_cls}">
              <div class="rule-title">{label}</div>
              <div class="rule-meta">
                {type_tag}
                <span class="tag" style="background:#1e2130;color:#94a3b8;">
                  scope: {scope}</span>
                {ov_badge}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Semantic analysis expander ─────────────────────────────────────────
        with st.expander("🧠 Semantic analysis detail"):
            for cname, info in sem.get("columns", {}).items():
                stype   = info.get("semantic_type", "unknown")
                role    = info.get("match_role", "")
                reason  = info.get("reasoning", "")
                primary = info.get("can_be_primary", False)
                css     = _STYPE_CSS.get(stype, "stype-other")
                p_badge = ('<span class="tag" style="background:#0c1a2e;color:#38bdf8;'
                           'font-size:.65rem;">primary ✓</span>' if primary else "")
                st.markdown(f"""<div class="sem-card">
                  <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;">
                    <span style="font-weight:600;font-size:.88rem;">{cname}</span>
                    <span class="tag {css}">{stype}</span>
                    {p_badge}
                    <span style="font-size:.75rem;color:#475569;margin-left:auto;">{role}</span>
                  </div>
                  <div class="sem-reason">{reason}</div>
                </div>""", unsafe_allow_html=True)

        # ── NL refinement ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Natural language refinement**")
        st.caption("Try: *remove the relevance rule*, *disable negative rules*, "
                   "*what rules were generated and why*")
        nl_input = st.text_input("Describe your refinement", placeholder="e.g. remove the relevance rule",
                                 key="nl_input")
        if st.button("Apply instruction") and nl_input.strip():
            msg   = nl_input.strip().lower()
            reply = ""
            updated_groups = list(rules.get("matchGroups", []))

            if any(w in msg for w in ("remove relevance", "delete relevance", "no relevance")):
                before = len(updated_groups)
                updated_groups = [g for g in updated_groups if g.get("type") != "relevance_based"]
                removed = before - len(updated_groups)
                reply = f"Removed {removed} relevance-based group(s)."

            elif any(w in msg for w in ("remove negative", "disable negative", "no negative")):
                before = len(updated_groups)
                updated_groups = [g for g in updated_groups if "negativeRule" not in g]
                removed = before - len(updated_groups)
                reply = f"Removed {removed} negative rule(s)."

            elif any(w in msg for w in ("remove suspect", "delete suspect", "no suspect")):
                before = len(updated_groups)
                updated_groups = [g for g in updated_groups if g.get("type") != "suspect"]
                removed = before - len(updated_groups)
                reply = f"Removed {removed} suspect group(s)."

            elif any(w in msg for w in ("how many", "what rules", "list rules", "show rules")):
                names = [g.get("label", g.get("uri", "?")) for g in updated_groups]
                reply = f"{len(names)} rules: " + "; ".join(names)

            elif any(w in msg for w in ("scope internal", "set scope internal")):
                for g in updated_groups:
                    g["scope"] = "INTERNAL"
                reply = "Set scope=INTERNAL on all groups."

            elif any(w in msg for w in ("scope all", "set scope all")):
                for g in updated_groups:
                    g["scope"] = "ALL"
                reply = "Set scope=ALL on all groups."

            else:
                reply = ("Not recognised. Try: 'remove the relevance rule', "
                         "'disable negative rules', 'set scope internal', 'list rules'.")

            if reply:
                rules["matchGroups"] = updated_groups
                st.session_state.match_rules = rules
                st.session_state.nl_history.append(
                    {"You": nl_input.strip(), "System": reply})
                st.rerun()

        if st.session_state.nl_history:
            with st.expander("Refinement history", expanded=True):
                for entry in reversed(st.session_state.nl_history):
                    st.markdown(f'<div class="nl-you">You: {entry["You"]}</div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div class="nl-sys">→ {entry["System"]}</div>',
                                unsafe_allow_html=True)

        st.markdown("---")
        if st.button("▶  Run match simulation", type="primary", use_container_width=True):
            with st.spinner(f"Simulating {len(df):,} records…"):
                counts = simulate_match_counts(df, rules)
            st.session_state.counts = counts
            st.session_state.step   = 5
            st.rerun()

# ── STEP 5: Simulate ──────────────────────────────────────────────────────────
elif st.session_state.step == 5:
    st.markdown('<div class="step-badge">STEP 05</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Simulation results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Match counts simulated against your full dataset</div>',
                unsafe_allow_html=True)

    df     = st.session_state.df
    rules  = st.session_state.match_rules
    counts = st.session_state.counts
    groups = rules.get("matchGroups", [])
    total  = len(df)

    total_mp    = sum(v.get("matching_profiles", 0) for v in counts.values())
    total_pairs = sum(v.get("matching_pairs",    0) for v in counts.values())
    n_auto      = sum(1 for g in groups if g.get("type") == "automatic")
    n_suspect   = sum(1 for g in groups if g.get("type") == "suspect")
    n_neg       = sum(1 for g in groups if "negativeRule" in g)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total profiles",    f"{total:,}")
    c2.metric("Profiles matched",  f"{total_mp:,}")
    c3.metric("Candidate pairs",   f"{total_pairs:,}")
    c4.metric("Auto-merge groups", n_auto)
    c5.metric("Suspect groups",    n_suspect)

    st.markdown("---")

    rows = []
    for g in groups:
        uri    = g.get("uri", "")
        label  = g.get("label", uri.split("/")[-1])
        rtype  = ("negativeRule" if "negativeRule" in g else g.get("type", "—"))
        c      = counts.get(uri, {})
        mp     = c.get("matching_profiles", 0)
        pairs  = c.get("matching_pairs",    0)
        clust  = c.get("largest_cluster",   0)
        method = c.get("method", "—")
        pct    = (mp / total * 100) if total else 0
        risk   = ("⚠️ High" if pct > 30 else "Medium" if pct > 10 else "Low")

        st.markdown(f"""<div class="rule-card {'neg' if rtype=='negativeRule' else ''}">
          <div class="rule-title">{label}</div>
          <div class="rule-meta">
            <span class="tag {'tag-neg' if rtype=='negativeRule' else 'tag-auto' if rtype=='automatic' else 'tag-suspect'}">{rtype}</span>
            <span class="tag" style="background:#1e2130;color:#94a3b8;">{method}</span>
          </div>
          <div class="match-stats">
            <div class="ms-item">
              <span class="ms-val {'warn' if pct>30 else ''}">{mp:,}{' ⚠️' if pct>30 else ''}</span>
              <span class="ms-lbl">profiles ({pct:.1f}%)</span>
            </div>
            <div class="ms-item">
              <span class="ms-val">{pairs:,}</span>
              <span class="ms-lbl">candidate pairs</span>
            </div>
            <div class="ms-item">
              <span class="ms-val">{clust:,}</span>
              <span class="ms-lbl">largest cluster</span>
            </div>
            <div class="ms-item">
              <span class="ms-val" style="color:{'#fb923c' if 'High' in risk else '#64748b'}">{risk}</span>
              <span class="ms-lbl">over-match risk</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        rows.append({
            "Match group":       label,
            "Type":              rtype,
            "Matching profiles": mp,
            "% of total":        f"{pct:.1f}%",
            "Candidate pairs":   pairs,
            "Largest cluster":   clust,
            "Method":            method,
            "Over-match risk":   risk,
        })

    st.markdown("---")
    if st.button("▶  Proceed to export", type="primary", use_container_width=True):
        st.session_state.step = 6
        st.rerun()

# ── STEP 6: Export ────────────────────────────────────────────────────────────
elif st.session_state.step == 6:
    st.markdown('<div class="step-badge">STEP 06</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Review & export</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download match rules for Reltio or Semarchy</div>',
                unsafe_allow_html=True)

    rules       = st.session_state.match_rules
    sem         = st.session_state.get("semantic", {})
    counts      = st.session_state.get("counts", {})
    fixes       = st.session_state.dq_fixes
    df          = st.session_state.df
    entity_type = st.session_state.entity_type
    groups      = rules.get("matchGroups", [])
    total       = len(df)
    ts          = datetime.now().strftime("%Y%m%d_%H%M")

    # Summary
    c1, c2 = st.columns(2)
    with c1:
        applied       = int(fixes["Apply"].sum())
        recs_fixed    = int(fixes[fixes["Apply"]]["Records"].sum())
        total_mp      = sum(v.get("matching_profiles", 0) for v in counts.values())
        total_pairs   = sum(v.get("matching_pairs",    0) for v in counts.values())
        st.markdown("**Pipeline summary**")
        st.markdown(f"- Entity type: **{entity_type}**")
        st.markdown(f"- Total records: **{total:,}**")
        st.markdown(f"- DQ fixes applied: **{applied}** ({recs_fixed:,} records)")
        st.markdown(f"- Match groups generated: **{len(groups)}**")
        st.markdown(f"- Profiles matched: **{total_mp:,}** across all rules")
        st.markdown(f"- Candidate pairs: **{total_pairs:,}**")

    with c2:
        st.markdown("**Match groups**")
        for g in groups:
            rtype = ("negativeRule" if "negativeRule" in g else g.get("type", "—"))
            label = g.get("label", g.get("uri", "?").split("/")[-1])
            icon  = {"automatic": "⚡", "suspect": "🔍",
                     "relevance_based": "⚖️", "negativeRule": "🚫"}.get(rtype, "•")
            st.markdown(f"{icon} {label}")

    st.markdown("---")

    # ── Export cards ──────────────────────────────────────────────────────────

    # 1. Reltio matchGroups JSON
    reltio_json = json.dumps(rules, indent=2)
    st.markdown('<div class="export-card">', unsafe_allow_html=True)
    st.markdown('<div class="export-title">🔷 Reltio — matchGroups JSON</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="export-sub">Drop this JSON into your Reltio tenant metadata '
        'configuration under the entity type definition. Contains all match groups '
        'including automatic, suspect, relevance-based and negative rules.</div>',
        unsafe_allow_html=True)
    st.code(reltio_json[:800] + ("\n..." if len(reltio_json) > 800 else ""),
            language="json")
    st.download_button(
        "⬇️  Download matchGroups.json",
        data=reltio_json,
        file_name=f"{entity_type}_matchGroups_{ts}.json",
        mime="application/json",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Semarchy YAML
    semarchy_yaml = reltio_to_semarchy_yaml(rules, entity_type)
    st.markdown('<div class="export-card">', unsafe_allow_html=True)
    st.markdown('<div class="export-title">🔶 Semarchy — Entity YAML (matcher block)</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="export-sub">Paste this YAML block inside your Semarchy entity design '
        'file. Uses SemQL match conditions with exact and fuzzy (SEM_EDIT_DISTANCE_SIMILARITY) '
        'logic. Match scores are set per Semarchy conventions: automatic→100, suspect→85, '
        'relevance→70. Negative rules are excluded — Semarchy handles prevention via merge '
        'thresholds.</div>',
        unsafe_allow_html=True)
    st.code(semarchy_yaml[:800] + ("\n..." if len(semarchy_yaml) > 800 else ""),
            language="yaml")
    st.download_button(
        "⬇️  Download semarchy_matcher.yaml",
        data=semarchy_yaml,
        file_name=f"{entity_type}_semarchy_matcher_{ts}.yaml",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Match counts CSV
    if counts:
        rows_csv = []
        for uri, c in counts.items():
            glabel = next((g.get("label", uri) for g in groups if g.get("uri") == uri), uri)
            pct    = (c["matching_profiles"] / total * 100) if total else 0
            rows_csv.append({
                "Match group":       glabel,
                "Type":              next((g.get("type", "negativeRule")
                                           for g in groups if g.get("uri") == uri), "—"),
                "Matching profiles": c["matching_profiles"],
                "% of total":        f"{pct:.1f}%",
                "Candidate pairs":   c["matching_pairs"],
                "Largest cluster":   c["largest_cluster"],
                "Method":            c["method"],
            })
        counts_csv = pd.DataFrame(rows_csv).to_csv(index=False)

        st.markdown('<div class="export-card">', unsafe_allow_html=True)
        st.markdown('<div class="export-title">📊 Match simulation results — CSV</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="export-sub">Per-rule match counts, candidate pairs, and '
            'largest cluster sizes simulated against your uploaded dataset.</div>',
            unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download match_counts.csv",
            data=counts_csv,
            file_name=f"{entity_type}_match_counts_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Semantic analysis JSON
    if sem:
        sem_json = json.dumps(sem, indent=2)
        st.markdown('<div class="export-card">', unsafe_allow_html=True)
        st.markdown('<div class="export-title">🧠 Semantic analysis — JSON</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="export-sub">The LLM\'s full column classification — semantic types, '
            'reasoning, match roles, and duplicate behaviour. Useful for auditing rule '
            'decisions or handing off to a Reltio/Semarchy implementation team.</div>',
            unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download semantic_analysis.json",
            data=sem_json,
            file_name=f"{entity_type}_semantic_analysis_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Start new iteration", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
