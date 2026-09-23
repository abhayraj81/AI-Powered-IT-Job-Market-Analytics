"""
skill_gap.py
============
Skill Gap Analyzer for the IT Job Market Analytics project.

Methodology:
    1. Build a "required skill profile" for the selected IT category or
       specific job title from the dataset.
    2. Normalise user's input skills using the same mapping as preprocessing.
    3. Compare user skills vs required profile to produce:
       - Matched skills
       - Missing skills (priority-ordered by demand frequency)
       - Skill coverage %
       - Recommended certifications for the target role

Scoring:
    Coverage (%) = |Matched| / |Required Profile| × 100

Important:  ALL skill requirements come directly from the dataset.
            No requirements are invented.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import normalize_skill

# ---------------------------------------------------------------------------
# Threshold: a skill is "required" if it appears in ≥ this % of roles
# in the target category.  Lowered vs previous dataset because the IT dataset
# has ~6 skills per role on average (vs 18 for the previous dataset).
# ---------------------------------------------------------------------------
MIN_SUPPORT = 0.08   # 8% of roles in the category


def get_category_skill_profile(df: pd.DataFrame,
                                category: str,
                                min_support: float = MIN_SUPPORT,
                                top_n: int = 50) -> pd.DataFrame:
    """
    Build a skill profile for an IT category.
    Returns a DataFrame: skill | count | frequency | rank
    """
    sub = df[df["category"].str.lower() == category.lower()]
    if len(sub) == 0:
        return pd.DataFrame(columns=["skill", "count", "frequency", "rank"])

    total_roles = len(sub)
    counts = Counter(s for skills in sub["skills"] for s in skills)

    records = [
        {"skill": skill, "count": cnt, "frequency": round(cnt / total_roles * 100, 2)}
        for skill, cnt in counts.items()
        if cnt / total_roles >= min_support
    ]
    profile = (
        pd.DataFrame(records)
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    profile["rank"] = profile.index + 1
    return profile.head(top_n)


def get_title_skill_profile(df: pd.DataFrame, job_title: str) -> pd.DataFrame:
    """
    Return all skills for a specific job title.
    Uses exact match first; falls back to substring.
    """
    sub = df[df["job_title"].str.lower() == job_title.lower()]
    if sub.empty:
        sub = df[df["job_title"].str.lower().str.contains(job_title.lower(), na=False)]
    if sub.empty:
        return pd.DataFrame(columns=["skill", "count", "frequency", "rank"])

    total = len(sub)
    counts = Counter(s for skills in sub["skills"] for s in skills)
    records = [
        {"skill": skill, "count": cnt,
         "frequency": round(cnt / total * 100, 2)}
        for skill, cnt in counts.items()
    ]
    profile = (
        pd.DataFrame(records)
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    profile["rank"] = profile.index + 1
    return profile


def get_category_cert_profile(df: pd.DataFrame, category: str,
                               top_n: int = 10) -> pd.DataFrame:
    """Return top certifications recommended for an IT category."""
    sub = df[df["category"].str.lower() == category.lower()]
    if sub.empty:
        return pd.DataFrame(columns=["certification", "count"])
    counts = Counter(c for certs in sub["certifications"] for c in certs)
    records = [{"certification": c, "count": cnt}
               for c, cnt in counts.most_common(top_n)]
    return pd.DataFrame(records)


def parse_user_skills(raw_input: str) -> list:
    """Parse and normalise comma-separated user skill input."""
    if not raw_input or not raw_input.strip():
        return []
    parts = [p.strip() for p in raw_input.split(",") if p.strip()]
    normalised = [normalize_skill(p) for p in parts]
    seen: set = set()
    result = []
    for s in normalised:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def analyze_skill_gap(user_skills: list, profile: pd.DataFrame) -> dict:
    """
    Compare user skills against the IT role skill profile.

    Returns a dict:
        matched_skills    – skills the user has that are in the profile
        missing_skills    – profile skills not in user's skill set (freq order)
        coverage_pct      – Skill Coverage %
        profile_size      – total skills in the required profile
        extra_skills      – user skills outside the required profile
        priority_missing  – top-10 missing skills to learn first
    """
    if profile.empty:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "coverage_pct": 0.0,
            "profile_size": 0,
            "extra_skills": user_skills,
            "priority_missing": [],
            "error": "No skill profile found for this role. "
                     "The category may have insufficient data.",
        }

    user_set = set(user_skills)
    required_set = set(profile["skill"].tolist())

    matched = sorted(user_set & required_set)
    missing = profile[~profile["skill"].isin(user_set)]["skill"].tolist()
    extra = sorted(user_set - required_set)
    coverage_pct = round(len(matched) / len(required_set) * 100, 2) if required_set else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "priority_missing": missing[:10],
        "coverage_pct": coverage_pct,
        "profile_size": len(required_set),
        "extra_skills": extra,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_skill_coverage_gauge(coverage_pct: float):
    """Gauge chart for skill coverage."""
    color = "#3b82d4" if coverage_pct >= 70 else ("#f59e0b" if coverage_pct >= 40 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=coverage_pct,
        title={"text": "Skill Coverage (%)"},
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


def plot_missing_skills_priority(profile: pd.DataFrame,
                                  missing: list, top_n: int = 15):
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
    sub = profile[profile["skill"].isin(matched)].sort_values("frequency", ascending=False)
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
        f"### Skill Gap Analysis — Target Role: {target_role}",
        "",
        f"**Skill Coverage:** {result['coverage_pct']}%  "
        f"({len(result['matched_skills'])} matched / {result['profile_size']} required)",
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
