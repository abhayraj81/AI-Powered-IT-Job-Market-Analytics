"""
NLP components for the AI-Powered IT Job Market Analytics project.

Provides:
- Text cleaning / tokenisation that KEEPS technical terms (C#, C++, R, Go, AI, ML,
  S3, CI/CD, .NET, Node.js ...). The old cleaner deleted digits, symbols and every
  token of <= 2 characters, which erased exactly the terms that matter in IT text.
- TF-IDF vectorisation on job descriptions
- Cosine-similarity job similarity
- Skill-level (not word-level) role recommendation
- N-gram analysis
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import skill_key


# ---------------------------------------------------------------------------
# Stopwords (single shared list for the whole project)
# ---------------------------------------------------------------------------

_BASE_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "than",
    "that", "this", "these", "those", "it", "its", "we", "our", "you",
    "your", "he", "she", "they", "their", "them", "who", "which", "what",
    "all", "each", "other", "such", "same", "more", "most", "also",
    "then", "there", "when", "where", "why", "how", "if", "while",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "including", "without", "must", "use", "using",
    "work", "year", "years", "experience", "related", "strong", "required",
    "knowledge", "skills", "skill", "position", "team", "role", "new",
    "ensure", "ensuring", "support", "provide", "develop", "management", "key",
    "job", "business", "within", "across", "well", "able", "ability",
    "excellent", "responsible", "responsibilities", "equivalent", "preferred",
    "plus", "least", "one", "two", "three", "etc", "any", "some", "us", "s",
    "requires", "expertise", "various", "overseeing",
}


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """Base list plus NLTK English stopwords when available (cached)."""
    words = set(_BASE_STOPWORDS)
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            words |= set(stopwords.words("english"))
        except LookupError:
            if nltk.download("stopwords", quiet=True):
                words |= set(stopwords.words("english"))
    except Exception:  # offline / nltk missing - the base list is enough
        pass
    return frozenset(words)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Multi-character tech terms are rewritten to plain tokens BEFORE tokenising.
_TECH_SUBS = [
    (re.compile(r"(?<![a-z0-9])c\+\+"), " cpp "),
    (re.compile(r"(?<![a-z0-9])c#"), " csharp "),
    (re.compile(r"(?<![a-z0-9])f#"), " fsharp "),
    (re.compile(r"asp\.net"), " aspnet "),
    (re.compile(r"\.net\b"), " dotnet "),
    (re.compile(r"node\.?js"), " nodejs "),
    (re.compile(r"react\.?js"), " reactjs "),
    (re.compile(r"vue\.?js"), " vuejs "),
    (re.compile(r"next\.?js"), " nextjs "),
    (re.compile(r"ci\s*/\s*cd"), " cicd "),
    (re.compile(r"ai\s*/\s*ml"), " ai ml "),
    (re.compile(r"pl\s*/\s*sql"), " plsql "),
    (re.compile(r"t-sql"), " tsql "),
    (re.compile(r"\br&d\b"), " rnd "),
]

# Single letters that are real technologies (C, R).
_KEEP_SINGLE = {"c", "r"}


def tokenize(text, stop_words=None) -> list:
    """Lowercase, protect tech terms, split on non-alphanumerics, drop stopwords."""
    if not isinstance(text, str):
        return []
    sw = stop_words if stop_words is not None else get_stopwords()
    t = text.lower()
    for pattern, replacement in _TECH_SUBS:
        t = pattern.sub(replacement, t)
    tokens = re.findall(r"[a-z0-9]+", t)
    return [
        tok for tok in tokens
        if not tok.isdigit()
        and (len(tok) >= 2 or tok in _KEEP_SINGLE)
        and tok not in sw
    ]


def clean_for_nlp(text, stop_words=None) -> str:
    """Return the cleaned text as a space-joined token string."""
    return " ".join(tokenize(text, stop_words))


# The cleaned text is already tokenised, so the vectoriser just splits on spaces.
_TFIDF_TOKEN_PATTERN = r"\S+"


def build_tfidf_matrix(df: pd.DataFrame, text_col: str = "job_description",
                       max_features: int = 3000, ngram_range: tuple = (1, 2)):
    """Fit TF-IDF on `text_col`. Returns (vectoriser, matrix, cleaned_texts)."""
    sw = get_stopwords()
    cleaned = df[text_col].apply(lambda t: clean_for_nlp(t, sw))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        token_pattern=_TFIDF_TOKEN_PATTERN,
    )
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix, cleaned


# ---------------------------------------------------------------------------
# TF-IDF: top terms per category
# ---------------------------------------------------------------------------

def top_tfidf_terms_by_category(df: pd.DataFrame, vectorizer, tfidf_matrix,
                                n: int = 15) -> dict:
    """Return {category: [(term, mean_tfidf), ...]}."""
    feature_names = vectorizer.get_feature_names_out()
    categories = df["category"].to_numpy()
    result = {}
    for cat in pd.unique(categories):
        rows = np.where(categories == cat)[0]
        mean_scores = np.asarray(tfidf_matrix[rows].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:n]
        result[cat] = [(feature_names[i], round(float(mean_scores[i]), 4)) for i in top_idx]
    return result


def plot_tfidf_terms(tfidf_by_cat: dict, category: str):
    """Bar chart of top TF-IDF terms for one category."""
    terms, scores = zip(*tfidf_by_cat[category])
    df_plot = pd.DataFrame({"Term": list(terms), "TF-IDF Score": list(scores)})
    fig = px.bar(
        df_plot.sort_values("TF-IDF Score"), x="TF-IDF Score", y="Term",
        orientation="h", color="TF-IDF Score",
        color_continuous_scale="Turbo",
        title=f"Top TF-IDF Terms - {category}",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# Job similarity
# ---------------------------------------------------------------------------

def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """Cosine similarity matrix (n_jobs x n_jobs)."""
    return cosine_similarity(tfidf_matrix)


def find_similar_jobs(df: pd.DataFrame, sim_matrix: np.ndarray,
                      job_idx: int, top_n: int = 5) -> pd.DataFrame:
    """Top-n most similar roles to the role at row position `job_idx`."""
    scores = sim_matrix[job_idx]
    order = [i for i in scores.argsort()[::-1] if i != job_idx][:top_n]
    rows = [{
        "job_title": df.iloc[i]["job_title"],
        "category": df.iloc[i]["category"],
        "similarity_score": round(float(scores[i]), 4),
    } for i in order]
    return pd.DataFrame(rows, columns=["job_title", "category", "similarity_score"])


# ---------------------------------------------------------------------------
# Skill-based role recommendation
# Each SKILL is one token ("ci/cd", "machine learning", "c#" stay intact), so
# nothing is split, stemmed or dropped. Matching is case-insensitive.
# ---------------------------------------------------------------------------

def _pipe_tokenizer(doc: str) -> list:
    """Module-level (picklable) tokeniser: 'a|b c|d' -> ['a', 'b c', 'd']."""
    return [t for t in doc.split("|") if t]


def build_skill_tfidf(df: pd.DataFrame):
    """Fit a skill-level TF-IDF index. Returns (vectoriser, matrix)."""
    docs = ["|".join(skill_key(s) for s in skills) for skills in df["skills"]]
    vectorizer = TfidfVectorizer(tokenizer=_pipe_tokenizer, token_pattern=None,
                                 lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


_REC_COLUMNS = ["job_title", "category", "similarity_score", "matched_skills", "role_skills"]


def recommend_jobs_by_skills(df: pd.DataFrame, user_skills: list, vectorizer,
                             matrix, top_n: int = 10) -> pd.DataFrame:
    """Roles whose skill sets best match `user_skills` (cosine on skill TF-IDF)."""
    user_keys = {skill_key(s) for s in user_skills}
    if not user_keys:
        return pd.DataFrame(columns=_REC_COLUMNS)

    query = vectorizer.transform(["|".join(sorted(user_keys))])
    scores = cosine_similarity(query, matrix).ravel()

    rows = []
    for i in scores.argsort()[::-1]:
        if scores[i] <= 0 or len(rows) >= top_n:
            break
        role_keys = {skill_key(s) for s in df.iloc[i]["skills"]}
        rows.append({
            "job_title": df.iloc[i]["job_title"],
            "category": df.iloc[i]["category"],
            "similarity_score": round(float(scores[i]), 4),
            "matched_skills": len(user_keys & role_keys),
            "role_skills": len(role_keys),
        })
    return pd.DataFrame(rows, columns=_REC_COLUMNS)


# ---------------------------------------------------------------------------
# N-gram analysis
# ---------------------------------------------------------------------------

def extract_ngrams(texts: pd.Series, n: int = 2, top_k: int = 30) -> pd.DataFrame:
    """Top-k n-grams (default bigrams). Returns columns: ngram, frequency."""
    sw = get_stopwords()
    counter: Counter = Counter()
    for text in texts.dropna():
        tokens = tokenize(str(text), sw)
        counter.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return pd.DataFrame(counter.most_common(top_k), columns=["ngram", "frequency"])


def plot_ngrams(df_ng: pd.DataFrame, title: str = "Top Bigrams in Job Descriptions"):
    """Horizontal bar chart of n-gram frequencies."""
    fig = px.bar(
        df_ng.sort_values("frequency"), x="frequency", y="ngram",
        orientation="h", color="frequency",
        color_continuous_scale="Purples",
        title=title, text="frequency",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig
