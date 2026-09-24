"""
EDA and visualisation functions for the IT Job Market Analytics project.

Accepts the cleaned DataFrame from preprocessing.load_and_clean().
Returns Plotly figures, Matplotlib figures, or computed DataFrames.

Notes
 - Categories are INFERRED from job titles (see preprocessing.CATEGORY_RULES).
 - Matplotlib figures returned from here must be closed by the caller
   (plt.close(fig)) after they are displayed.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud

from src.nlp import tokenize

PLOTLY_PALETTE = px.colors.qualitative.Safe


# ---------------------------------------------------------------------------
# 1. Category overview
# ---------------------------------------------------------------------------

def category_distribution(df: pd.DataFrame):
    """Bar + pie charts for the inferred category distribution."""
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


# ---------------------------------------------------------------------------
# 2. Job title analysis
# ---------------------------------------------------------------------------

def top_job_titles(df: pd.DataFrame, n: int = 25):
    """All titles are unique (after dedup), so this ranks them by skill count."""
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
    """Job titles in a category (sorted by skill count)."""
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


# ---------------------------------------------------------------------------
# 3. Skill analysis
# ---------------------------------------------------------------------------

def get_skill_counts(df: pd.DataFrame) -> pd.Series:
    """skill -> number of roles listing it."""
    all_skills = [s for skills in df["skills"] for s in skills]
    return pd.Series(Counter(all_skills), dtype="int64").sort_values(ascending=False)


def top_skills_overall(df: pd.DataFrame, n: int = 25):
    """Horizontal bar of the most demanded skills overall."""
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
    counts = (pd.Series(Counter(s for skills in sub["skills"] for s in skills), dtype="int64")
              .sort_values(ascending=False).head(n).reset_index())
    counts.columns = ["Skill", "Frequency"]

    fig = px.bar(
        counts.sort_values("Frequency"), x="Frequency", y="Skill",
        orientation="h", color="Frequency",
        color_continuous_scale="Plasma",
        title=f"Top {n} Skills - {category}",
        text="Frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig, counts


def skills_per_category_heatmap(df: pd.DataFrame, top_n_skills: int = 30):
    """Normalised skill-category heatmap (% of roles in category needing skill)."""
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
    ax.set_title("Skill-Category Heatmap (% of roles requiring each skill)", fontsize=13)
    ax.set_xlabel("Skill")
    ax.set_ylabel("Category")
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", fontsize=8)
    fig.tight_layout()
    return fig, heat_norm


# ---------------------------------------------------------------------------
# 4. Skill co-occurrence
# ---------------------------------------------------------------------------

def skill_cooccurrence(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Most common skill pairs among the top-n skills."""
    top_skills = set(get_skill_counts(df).head(top_n).index)
    pair_counts: Counter = Counter()
    for skills in df["skills"]:
        filtered = [s for s in skills if s in top_skills]
        for a, b in combinations(sorted(set(filtered)), 2):
            pair_counts[(a, b)] += 1

    records = [{"Skill A": a, "Skill B": b, "Co-occurrences": c}
               for (a, b), c in pair_counts.most_common(50)]
    return pd.DataFrame(records, columns=["Skill A", "Skill B", "Co-occurrences"])


def skill_cooccurrence_heatmap(df: pd.DataFrame, top_n: int = 20):
    """Interactive Plotly heatmap for top-n skill co-occurrences."""
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


# ---------------------------------------------------------------------------
# 5. Category-distinctive skills
# ---------------------------------------------------------------------------

def category_unique_skills(df: pd.DataFrame, top_n: int = 10, min_count: int = 3) -> dict:
    """
    Skills that are both DISTINCTIVE for a category and COMMON inside it.

    For a skill in a category:
        precision = roles in category with skill / all roles with skill
        recall    = roles in category with skill / roles in category
        score     = F1(precision, recall) x 100

    The previous score was precision only, so a skill seen 2 times in total
    (both in one category) scored 100% and beat genuinely defining skills.
    Skills seen in fewer than `min_count` roles of the category are ignored.
    Returns {category: [(skill, score), ...]}.
    """
    global_counts = get_skill_counts(df)
    result = {}
    for cat, sub in df.groupby("category"):
        n_roles = len(sub)
        cat_counts = Counter(s for skills in sub["skills"] for s in skills)
        scored = []
        for skill, cnt in cat_counts.items():
            if cnt < min_count:
                continue
            precision = cnt / global_counts[skill]
            recall = cnt / n_roles
            f1 = 2 * precision * recall / (precision + recall)
            scored.append((skill, round(f1 * 100, 1)))
        scored.sort(key=lambda x: (-x[1], x[0]))
        result[cat] = scored[:top_n]
    return result


# ---------------------------------------------------------------------------
# 6. Certification analysis
# ---------------------------------------------------------------------------

def get_cert_counts(df: pd.DataFrame) -> pd.Series:
    """certification -> frequency across all roles."""
    all_certs = [c for certs in df["certifications"] for c in certs]
    return pd.Series(Counter(all_certs), dtype="int64").sort_values(ascending=False)


def top_certifications_overall(df: pd.DataFrame, n: int = 20):
    """Bar chart of the most referenced certifications."""
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
    counts = (pd.Series(Counter(c for certs in sub["certifications"] for c in certs), dtype="int64")
              .sort_values(ascending=False).head(n).reset_index())
    counts.columns = ["Certification", "Frequency"]

    fig = px.bar(
        counts.sort_values("Frequency"), x="Frequency", y="Certification",
        orientation="h", color="Frequency",
        color_continuous_scale="Purples",
        title=f"Top {n} Certifications - {category}",
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


# ---------------------------------------------------------------------------
# 7. Word cloud (built from whole skill / certification names)
# ---------------------------------------------------------------------------

def generate_wordcloud(df: pd.DataFrame, kind: str = "skills", category: str | None = None):
    """
    Word cloud of whole skill (or certification) names weighted by frequency.

    The old version joined everything into one string, so multi-word skills such
    as "Machine Learning" were split into separate words.
    kind: "skills" or "certifications".
    """
    column = "certifications" if kind.startswith("cert") else "skills"
    sub = df[df["category"] == category] if category else df
    freqs = Counter(item for items in sub[column] for item in items)
    if not freqs:
        return None

    wc = WordCloud(
        width=900, height=380, background_color="white",
        colormap="viridis", max_words=120,
    ).generate_from_frequencies(freqs)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    label = "Certification" if column == "certifications" else "Skill"
    ax.set_title(f"{label} Word Cloud - {category}" if category
                 else f"{label} Word Cloud - All IT Roles", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Job description word frequency (uses the shared NLP tokeniser)
# ---------------------------------------------------------------------------

def top_description_words(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Top words in job descriptions (shared stopwords; keeps terms like C#, S3, CI/CD)."""
    counter: Counter = Counter()
    for desc in df["job_description"].dropna():
        counter.update(tokenize(desc))
    return pd.DataFrame(counter.most_common(n), columns=["Word", "Frequency"])


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


# ---------------------------------------------------------------------------
# 9. Skill count distribution
# ---------------------------------------------------------------------------

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
