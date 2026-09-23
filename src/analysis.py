"""
analysis.py
===========
EDA and visualisation functions for the IT Job Market Analytics project.

Accepts the cleaned DataFrame from preprocessing.load_and_clean().
Returns Plotly figures, Matplotlib figures, or computed DataFrames.

New vs previous dataset:
 - Categories are INFERRED, not pre-labelled.
 - Dataset is IT-only: all roles are IT specialisations.
 - Certifications column exists and is analysed separately.
"""

from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud

PLOTLY_PALETTE = px.colors.qualitative.Safe


# ============================================================================
# 1.  Category overview
# ============================================================================

def category_distribution(df: pd.DataFrame):
    """Bar + pie charts for inferred category distribution."""
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    bar = px.bar(
        counts, x="Category", y="Count", color="Category",
        color_discrete_sequence=PLOTLY_PALETTE,
        title="IT Job Roles by Inferred Category",
        text="Count",
    )
    bar.update_traces(textposition="outside")
    bar.update_layout(showlegend=False, xaxis_tickangle=-30,
                      xaxis_title="Category", yaxis_title="Number of Roles")

    pie = px.pie(
        counts, names="Category", values="Count",
        color_discrete_sequence=PLOTLY_PALETTE,
        title="Category Distribution (%)",
        hole=0.35,
    )
    pie.update_traces(textposition="inside", textinfo="percent+label")
    return bar, pie, counts


# ============================================================================
# 2.  Job Title analysis
# ============================================================================

def top_job_titles(df: pd.DataFrame, n: int = 25):
    """All titles are unique (after dedup), so this shows top by skill count."""
    top = df.nlargest(n, "skill_count")[["job_title", "category", "skill_count"]]
    fig = px.bar(
        top.sort_values("skill_count"), x="skill_count", y="job_title",
        orientation="h", color="category",
        color_discrete_sequence=PLOTLY_PALETTE,
        title=f"Top {n} IT Roles by Number of Required Skills",
        labels={"skill_count": "Skills Listed", "job_title": "Job Role"},
        text="skill_count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig, top


def titles_by_category(df: pd.DataFrame, category: str, n: int = 15):
    """All job titles in a category (sorted by skill count)."""
    sub = df[df["category"] == category].nlargest(n, "skill_count")
    fig = px.bar(
        sub.sort_values("skill_count"), x="skill_count", y="job_title",
        orientation="h", color="skill_count",
        color_continuous_scale="Blues",
        title=f"IT Roles in '{category}' (by required skills)",
        labels={"skill_count": "Skills Listed", "job_title": "Job Role"},
        text="skill_count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig, sub


# ============================================================================
# 3.  Skill analysis
# ============================================================================

def get_skill_counts(df: pd.DataFrame) -> pd.Series:
    """Return skill → frequency across all job roles."""
    all_skills = [s for skills in df["skills"] for s in skills]
    return pd.Series(Counter(all_skills)).sort_values(ascending=False)


def top_skills_overall(df: pd.DataFrame, n: int = 25):
    """Horizontal bar of most demanded skills overall."""
    skill_counts = get_skill_counts(df).head(n).reset_index()
    skill_counts.columns = ["Skill", "Frequency"]

    fig = px.bar(
        skill_counts.sort_values("Frequency"), x="Frequency", y="Skill",
        orientation="h", color="Frequency",
        color_continuous_scale="Viridis",
        title=f"Top {n} Most In-Demand IT Skills",
        text="Frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig, skill_counts


def top_skills_by_category(df: pd.DataFrame, category: str, n: int = 20):
    """Top skills within a category."""
    sub = df[df["category"] == category]
    all_skills = [s for skills in sub["skills"] for s in skills]
    counts = pd.Series(Counter(all_skills)).sort_values(ascending=False).head(n).reset_index()
    counts.columns = ["Skill", "Frequency"]

    fig = px.bar(
        counts.sort_values("Frequency"), x="Frequency", y="Skill",
        orientation="h", color="Frequency",
        color_continuous_scale="Plasma",
        title=f"Top {n} Skills — {category}",
        text="Frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig, counts


def skills_per_category_heatmap(df: pd.DataFrame, top_n_skills: int = 30):
    """Normalised skill–category heatmap (% of roles in category needing skill)."""
    global_top = get_skill_counts(df).head(top_n_skills).index.tolist()
    cat_order = df["category"].value_counts().index.tolist()

    rows = []
    for cat in cat_order:
        sub = df[df["category"] == cat]
        cat_counts = Counter(s for skills in sub["skills"] for s in skills)
        rows.append({skill: cat_counts.get(skill, 0) for skill in global_top})

    heat_df = pd.DataFrame(rows, index=cat_order)
    cat_job_counts = df["category"].value_counts()
    heat_norm = heat_df.div(cat_job_counts[heat_df.index], axis=0) * 100

    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(
        heat_norm, ax=ax, cmap="YlOrRd", linewidths=0.3,
        cbar_kws={"label": "% of roles requiring skill"},
        xticklabels=True,
    )
    ax.set_title(f"Skill–Category Heatmap (% of roles requiring each skill)", fontsize=13)
    ax.set_xlabel("Skill")
    ax.set_ylabel("Category")
    plt.xticks(rotation=50, ha="right", fontsize=8)
    plt.tight_layout()
    return fig, heat_norm


# ============================================================================
# 4.  Skill co-occurrence
# ============================================================================

def skill_cooccurrence(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Return DataFrame of most common skill pairs."""
    top_skills = set(get_skill_counts(df).head(top_n).index)
    pair_counts: Counter = Counter()
    for skills in df["skills"]:
        filtered = [s for s in skills if s in top_skills]
        for a, b in combinations(sorted(set(filtered)), 2):
            pair_counts[(a, b)] += 1

    records = [{"Skill A": a, "Skill B": b, "Co-occurrences": c}
               for (a, b), c in pair_counts.most_common(50)]
    return pd.DataFrame(records)


def skill_cooccurrence_heatmap(df: pd.DataFrame, top_n: int = 20):
    """Interactive Plotly heatmap for top-N skill co-occurrences."""
    top_skills = get_skill_counts(df).head(top_n).index.tolist()
    matrix = pd.DataFrame(0, index=top_skills, columns=top_skills)
    for skills in df["skills"]:
        filtered = [s for s in skills if s in top_skills]
        for a, b in combinations(set(filtered), 2):
            matrix.loc[a, b] += 1
            matrix.loc[b, a] += 1

    fig = px.imshow(
        matrix, color_continuous_scale="Blues",
        title=f"Skill Co-occurrence Heatmap (Top {top_n} IT Skills)",
        aspect="auto", labels={"color": "Co-occurrences"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig, matrix


# ============================================================================
# 5.  Category-distinctive skills
# ============================================================================

def category_unique_skills(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Relative concentration score per category (TF-IDF-style)."""
    global_counts = get_skill_counts(df)
    result = {}
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        cat_counts = Counter(s for skills in sub["skills"] for s in skills)
        scores = {
            skill: (cnt / max(global_counts[skill], 1)) * 100
            for skill, cnt in cat_counts.items()
            if global_counts.get(skill, 0) > 1
        }
        result[cat] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return result


# ============================================================================
# 6.  Certification analysis (new: unique to this dataset)
# ============================================================================

def get_cert_counts(df: pd.DataFrame) -> pd.Series:
    """Return certification → frequency across all roles."""
    all_certs = [c for certs in df["certifications"] for c in certs]
    return pd.Series(Counter(all_certs)).sort_values(ascending=False)


def top_certifications_overall(df: pd.DataFrame, n: int = 20):
    """Bar chart of most referenced certifications."""
    cert_counts = get_cert_counts(df).head(n).reset_index()
    cert_counts.columns = ["Certification", "Frequency"]

    fig = px.bar(
        cert_counts.sort_values("Frequency"), x="Frequency", y="Certification",
        orientation="h", color="Frequency",
        color_continuous_scale="Teal",
        title=f"Top {n} Most Referenced IT Certifications",
        text="Frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig, cert_counts


def top_certifications_by_category(df: pd.DataFrame, category: str, n: int = 15):
    """Top certifications within a specific IT category."""
    sub = df[df["category"] == category]
    all_certs = [c for certs in sub["certifications"] for c in certs]
    counts = pd.Series(Counter(all_certs)).sort_values(ascending=False).head(n).reset_index()
    counts.columns = ["Certification", "Frequency"]

    fig = px.bar(
        counts.sort_values("Frequency"), x="Frequency", y="Certification",
        orientation="h", color="Frequency",
        color_continuous_scale="Purples",
        title=f"Top {n} Certifications — {category}",
        text="Frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig, counts


def cert_count_distribution(df: pd.DataFrame):
    """Histogram of certifications required per role."""
    fig = px.histogram(
        df, x="cert_count", nbins=20, color="category",
        color_discrete_sequence=PLOTLY_PALETTE,
        title="Certifications Required per IT Role",
        labels={"cert_count": "Number of Certifications"},
        barmode="overlay", opacity=0.7,
    )
    fig.update_layout(xaxis_title="Certifications per Role", yaxis_title="Number of Roles")
    return fig


# ============================================================================
# 7.  Word Cloud
# ============================================================================

def generate_wordcloud(df: pd.DataFrame, column: str = "skills_str",
                       category: str = None):
    """Word cloud from skills_str or certs_str, optionally filtered by category."""
    sub = df[df["category"] == category] if category else df
    text = " ".join(sub[column].dropna())
    if not text.strip():
        return None

    wc = WordCloud(
        width=900, height=380, background_color="white",
        colormap="viridis", max_words=120, collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    label = column.replace("_str", "").replace("_", " ").title()
    title = f"{label} Word Cloud — {category}" if category else f"{label} Word Cloud — All IT Roles"
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig


# ============================================================================
# 8.  Job Description word frequency
# ============================================================================

def top_description_words(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Top words in job descriptions (stopwords removed)."""
    import re
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))
    except ImportError:
        stop_words = set()

    extra_stop = {
        "ability", "also", "including", "must", "may", "will", "use", "using",
        "work", "year", "years", "experience", "related", "strong", "required",
        "knowledge", "skills", "skill", "position", "team", "role", "new",
        "ensure", "support", "provide", "develop", "management", "key",
        "business", "within", "across", "well", "able", "excellent",
        "responsible", "responsibilities", "requires", "expertise",
        "including", "various", "overseeing", "ensuring",
    }
    stop_words |= extra_stop

    all_words: Counter = Counter()
    for desc in df["job_description"].dropna():
        words = re.findall(r"[a-z]{3,}", desc.lower())
        all_words.update(w for w in words if w not in stop_words)

    return pd.DataFrame(all_words.most_common(n), columns=["Word", "Frequency"])


def plot_top_words(freq_df: pd.DataFrame):
    """Horizontal bar chart of top description words."""
    fig = px.bar(
        freq_df.sort_values("Frequency"), x="Frequency", y="Word",
        orientation="h", color="Frequency",
        color_continuous_scale="Teal",
        title="Top Terms in IT Job Descriptions",
        text="Frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


# ============================================================================
# 9.  Skill count distribution
# ============================================================================

def skill_count_distribution(df: pd.DataFrame):
    """Histogram of skill counts per IT role."""
    fig = px.histogram(
        df, x="skill_count", nbins=20, color="category",
        color_discrete_sequence=PLOTLY_PALETTE,
        title="Distribution of Skills per IT Role",
        labels={"skill_count": "Number of Skills"},
        barmode="overlay", opacity=0.75,
    )
    fig.update_layout(xaxis_title="Number of Skills Listed", yaxis_title="Number of Roles")
    return fig
