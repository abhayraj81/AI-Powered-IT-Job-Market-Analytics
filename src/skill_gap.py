"""
Skill Gap Analyzer for the IT Job Market Analytics project.

Methodology:
    1. Build a "required skill profile" for the selected IT category (or a
       specific job title) from the dataset.
    2. Normalise the user's skills and compare them CASE-INSENSITIVELY.
    3. Produce matched skills, missing skills (ordered by demand), coverage
       scores and recommended certifications.

Scoring:
    Simple coverage (%)   = |matched| / |profile| x 100
    Weighted coverage (%) = sum(demand of matched) / sum(demand of profile) x 100
    "Demand" is the % of roles in the category that list the skill, so knowing a
    skill that 80% of roles need counts far more than one that 8% need.
    `coverage_pct` (used by the gauge) is the WEIGHTED value.

Profile rules:
    A skill is "required" when it appears in >= MIN_SUPPORT of the category's
    roles AND in at least MIN_COUNT roles (so tiny categories cannot turn every
    one-off skill into a requirement). Categories with fewer than
    MIN_ROLES_FOR_ANALYSIS roles are not offered as gap-analysis targets.

All requirements come from the dataset; nothing is invented.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import parse_user_skills, skill_key  # noqa: F401  (re-exported)

MIN_SUPPORT = 0.08          # skill must appear in >= 8% of the category's roles
MIN_COUNT = 2               # ... and in at least 2 roles
MIN_ROLES_FOR_ANALYSIS = 15
EXCLUDED_CATEGORIES = {"Other IT"}

_PROFILE_COLUMNS = ["skill", "count", "frequency", "rank", "key"]


def eligible_categories(df: pd.DataFrame, min_roles: int = MIN_ROLES_FOR_ANALYSIS) -> list:
    """Categories with enough roles to give a meaningful skill profile."""
    counts = df["category"].value_counts()
    return sorted(c for c, n in counts.items() if n >= min_roles and c not in EXCLUDED_CATEGORIES)


def _skill_table(sub: pd.DataFrame, min_support: float = 0.0, min_count: int = 1) -> pd.DataFrame:
    total = len(sub)
    counts = Counter(s for skills in sub["skills"] for s in skills)
    records = [
        {"skill": s, "count": c, "frequency": round(c / total * 100, 2), "key": skill_key(s)}
        for s, c in counts.items()
        if c >= min_count and c / total >= min_support
    ]
    if not records:  # the old code crashed here (sort_values on a missing column)
        return pd.DataFrame(columns=_PROFILE_COLUMNS)
    table = (pd.DataFrame(records)
             .sort_values(["frequency", "skill"], ascending=[False, True])
             .reset_index(drop=True))
    table["rank"] = table.index + 1
    return table[_PROFILE_COLUMNS]


def get_category_skill_profile(df: pd.DataFrame, category: str,
                               min_support: float = MIN_SUPPORT,
                               min_count: int = MIN_COUNT,
                               top_n: int = 50) -> pd.DataFrame:
    """Skill profile for an IT category: skill | count | frequency | rank | key."""
    sub = df[df["category"].str.lower() == category.lower()]
    if sub.empty:
        return pd.DataFrame(columns=_PROFILE_COLUMNS)
    return _skill_table(sub, min_support, min_count).head(top_n)


def get_title_skill_profile(df: pd.DataFrame, job_title: str) -> pd.DataFrame:
    """Skills of one job title (exact match first, then plain-text substring)."""
    lowered = df["job_title"].str.lower()
    sub = df[lowered == job_title.lower()]
    if sub.empty:
        # regex=False: titles like "C++ Developer" or ".NET Developer" broke the regex search
        sub = df[lowered.str.contains(job_title.lower(), regex=False, na=False)]
    if sub.empty:
        return pd.DataFrame(columns=_PROFILE_COLUMNS)
    return _skill_table(sub)


def get_category_cert_profile(df: pd.DataFrame, category: str, top_n: int = 10) -> pd.DataFrame:
    """Top certifications for an IT category."""
    sub = df[df["category"].str.lower() == category.lower()]
    if sub.empty:
        return pd.DataFrame(columns=["certification", "count"])
    counts = Counter(c for certs in sub["certifications"] for c in certs)
    return pd.DataFrame([{"certification": c, "count": n} for c, n in counts.most_common(top_n)],
                        columns=["certification", "count"])


def analyze_skill_gap(user_skills: list, profile: pd.DataFrame) -> dict:
    """
    Compare user skills with a skill profile (case-insensitive).

    Returns: matched_skills, missing_skills, priority_missing (top 10),
             coverage_pct (demand-weighted), simple_coverage_pct,
             profile_size, extra_skills.
    """
    if profile.empty:
        return {
            "matched_skills": [], "missing_skills": [], "priority_missing": [],
            "coverage_pct": 0.0, "simple_coverage_pct": 0.0, "profile_size": 0,
            "extra_skills": list(user_skills),
            "error": "No skill profile found for this role. The category may have insufficient data.",
        }

    user_by_key = {skill_key(s): s for s in user_skills}
    keys = profile["key"] if "key" in profile else profile["skill"].map(skill_key)
    is_match = keys.isin(list(user_by_key))

    matched = profile.loc[is_match, "skill"].tolist()      # highest-demand first
    missing = profile.loc[~is_match, "skill"].tolist()     # highest-demand first
    extra = sorted(user_by_key[k] for k in set(user_by_key) - set(keys))

    total_demand = float(profile["frequency"].sum())
    weighted = float(profile.loc[is_match, "frequency"].sum()) / total_demand * 100 if total_demand else 0.0
    simple = len(matched) / len(profile) * 100

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "priority_missing": missing[:10],
        "coverage_pct": round(weighted, 2),
        "simple_coverage_pct": round(simple, 2),
        "profile_size": len(profile),
        "extra_skills": extra,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_skill_coverage_gauge(coverage_pct: float):
    """Gauge chart for (demand-weighted) skill coverage."""
    color = "#3b82d4" if coverage_pct >= 70 else ("#f59e0b" if coverage_pct >= 40 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=coverage_pct,
        title={"text": "Weighted Skill Coverage (%)"},
        delta={"reference": 70, "increasing": {"color": "#22c55e"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": "darkgray", "width": 3},
                          "thickness": 0.75, "value": 70},
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=280)
    return fig


def plot_gap_breakdown(matched: list, missing: list):
    """Donut chart: matched vs missing skills."""
    fig = go.Figure(go.Pie(
        labels=["Skills You Have", "Skills to Acquire"],
        values=[len(matched), len(missing)],
        marker={"colors": ["#22c55e", "#f87171"]},
        hole=0.45,
        textinfo="label+value+percent",
    ))
    fig.update_layout(title="Skill Match Breakdown", height=320)
    return fig


def plot_missing_skills_priority(profile: pd.DataFrame, missing: list, top_n: int = 15):
    """Horizontal bar of missing skills ordered by demand in the target role."""
    sub = profile[profile["skill"].isin(set(missing[:top_n]))].head(top_n)
    fig = px.bar(
        sub.sort_values("frequency"), x="frequency", y="skill",
        orientation="h", color="frequency",
        color_continuous_scale="Reds",
        title=f"Top {top_n} Skills to Learn (by Demand in Target Role)",
        labels={"frequency": "% of roles requiring skill", "skill": "Skill"},
        text="frequency",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def plot_matched_skills(profile: pd.DataFrame, matched: list):
    """Bar chart of matched skill demand."""
    if not matched:
        return go.Figure().update_layout(title="No Matched Skills")
    sub = profile[profile["skill"].isin(matched)]
    fig = px.bar(
        sub.sort_values("frequency"), x="frequency", y="skill",
        orientation="h", color="frequency",
        color_continuous_scale="Greens",
        title="Your Matched Skills (Demand in Target Role)",
        labels={"frequency": "% of roles requiring skill", "skill": "Skill"},
        text="frequency",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def plot_recommended_certs(cert_profile: pd.DataFrame, top_n: int = 8):
    """Bar chart of top certifications for the target role."""
    if cert_profile.empty:
        return go.Figure().update_layout(title="No certification data")
    sub = cert_profile.head(top_n)
    fig = px.bar(
        sub.sort_values("count"), x="count", y="certification",
        orientation="h", color="count",
        color_continuous_scale="Purples",
        title=f"Top {top_n} Recommended Certifications for Target Role",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def format_gap_summary(result: dict, target_role: str) -> str:
    """Plain-text summary of the skill gap analysis."""
    if result.get("error"):
        return f"⚠️  {result['error']}"

    lines = [
        f"### Skill Gap Analysis - Target Role: {target_role}",
        "",
        f"**Weighted coverage:** {result['coverage_pct']}%  "
        f"(simple: {result['simple_coverage_pct']}% = "
        f"{len(result['matched_skills'])} matched / {result['profile_size']} required)",
        "",
        "**✅ Matching Skills:**",
    ]
    for s in result["matched_skills"] or ["None"]:
        lines.append(f"  - {s}")

    lines += ["", "**❌ Top Skills to Learn:**"]
    for s in result["priority_missing"] or ["You already have all key skills! 🎉"]:
        lines.append(f"  - {s}")

    if result["extra_skills"]:
        lines += ["", "**➕ Your Additional Skills (Beyond Core Profile):**"]
        for s in result["extra_skills"][:10]:
            lines.append(f"  - {s}")

    return "\n".join(lines)
