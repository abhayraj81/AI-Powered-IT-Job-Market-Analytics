"""
NLP components for the AI-Powered Job Market Analytics project.

Provides:
- Text cleaning / tokenization
- TF-IDF vectorisation on job descriptions
- Cosine similarity – based job similarity / recommendation
- N-gram analysis
- Keyword extraction from job descriptions
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Stopword list (lightweight – avoids mandatory NLTK data download)
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
    "during", "including", "without", "must", "will", "use", "using",
    "work", "year", "years", "experience", "related", "strong", "required",
    "knowledge", "skills", "skill", "position", "team", "role", "new",
    "ensure", "support", "provide", "develop", "management", "key", "job",
    "business", "within", "across", "well", "able", "ability", "excellent",
    "responsible", "responsibilities", "equivalent", "preferred", "plus",
    "least", "one", "two", "three", "including", "etc", "any", "some",
    "other", "us", "s",
}


def _get_stopwords() -> set:
    """Return combined stopwords (base + optional NLTK)."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return _BASE_STOPWORDS | set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return _BASE_STOPWORDS | set(stopwords.words("english"))
    except ImportError:
        return _BASE_STOPWORDS


def clean_for_nlp(text: str, stop_words: set = None) -> str:
    """
    Clean a text string for NLP tasks:
    1. Lowercase
    2. Remove non-alphabetic characters
    3. Collapse whitespace
    4. Remove stopwords
    Returns cleaned string (space-joined tokens).
    """
    if not isinstance(text, str):
        return ""
    sw = stop_words if stop_words is not None else _get_stopwords()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2 and t not in sw]
    return " ".join(tokens)


def build_tfidf_matrix(df: pd.DataFrame, text_col: str = "job_description",
                       max_features: int = 3000,
                       ngram_range: tuple = (1, 2)):
    """
    Fit a TF-IDF vectoriser on `text_col` and return:
        (vectoriser, tfidf_matrix, cleaned_texts)
    """
    sw = _get_stopwords()
    cleaned = df[text_col].apply(lambda t: clean_for_nlp(t, sw))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,          # must appear in at least 2 docs
        max_df=0.90,       # ignore terms appearing in >90% of docs
        sublinear_tf=True, # log-normalised TF
    )
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix, cleaned


# ---------------------------------------------------------------------------
# TF-IDF: Top terms per category
# ---------------------------------------------------------------------------

def top_tfidf_terms_by_category(df: pd.DataFrame, vectorizer, tfidf_matrix,
                                 n: int = 15) -> dict:
    """
    For each category, aggregate TF-IDF scores and return top-n terms.
    Returns dict {category: [(term, score), ...]}
    """
    feature_names = vectorizer.get_feature_names_out()
    result = {}

    for cat in df["category"].unique():
        idx = df[df["category"] == cat].index
        # Mean TF-IDF score across all jobs in category
        cat_matrix = tfidf_matrix[idx]
        mean_scores = np.array(cat_matrix.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[::-1][:n]
        result[cat] = [(feature_names[i], round(float(mean_scores[i]), 4))
                       for i in top_idx]
    return result


def plot_tfidf_terms(tfidf_by_cat: dict, category: str):
    """Bar chart of top TF-IDF terms for one category."""
    terms, scores = zip(*tfidf_by_cat[category])
    df_plot = pd.DataFrame({"Term": list(terms), "TF-IDF Score": list(scores)})
    fig = px.bar(
        df_plot.sort_values("TF-IDF Score"), x="TF-IDF Score", y="Term",
        orientation="h", color="TF-IDF Score",
        color_continuous_scale="Turbo",
        title=f"Top TF-IDF Terms – {category}",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# Job Similarity / Recommendation
# ---------------------------------------------------------------------------

def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """Return cosine similarity matrix (n_jobs × n_jobs)."""
    return cosine_similarity(tfidf_matrix)


def find_similar_jobs(df: pd.DataFrame, sim_matrix: np.ndarray,
                      job_idx: int, top_n: int = 5) -> pd.DataFrame:
    """
    Given a job index, return the top_n most similar jobs.
    Returns a DataFrame with columns: job_title, category, similarity_score.
    """
    scores = sim_matrix[job_idx]
    # Exclude the job itself
    similar_idx = scores.argsort()[::-1]
    similar_idx = [i for i in similar_idx if i != job_idx][:top_n]

    rows = []
    for i in similar_idx:
        rows.append({
            "job_title": df.iloc[i]["job_title"],
            "category": df.iloc[i]["category"],
            "similarity_score": round(float(scores[i]), 4),
        })
    return pd.DataFrame(rows)


def recommend_jobs_by_skills(df: pd.DataFrame, user_skills: list[str],
                              vectorizer, tfidf_matrix,
                              top_n: int = 10) -> pd.DataFrame:
    """
    Given a list of user skills, find the most similar job postings using
    TF-IDF cosine similarity on the skills text.
    Returns a DataFrame of top_n recommended jobs.
    """
    query = " ".join(user_skills)
    sw = _get_stopwords()
    query_clean = clean_for_nlp(query, sw)
    query_vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_idx = scores.argsort()[::-1][:top_n]
    rows = []
    for i in top_idx:
        rows.append({
            "job_title": df.iloc[i]["job_title"],
            "category": df.iloc[i]["category"],
            "similarity_score": round(float(scores[i]), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# N-gram analysis
# ---------------------------------------------------------------------------

def extract_ngrams(texts: pd.Series, n: int = 2, top_k: int = 30) -> pd.DataFrame:
    """
    Extract top-k n-grams (default bigrams) from a Series of cleaned texts.
    Returns a DataFrame with columns: ngram, frequency.
    """
    sw = _get_stopwords()
    counter: Counter = Counter()

    for text in texts.dropna():
        cleaned = clean_for_nlp(str(text), sw)
        tokens = cleaned.split()
        grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counter.update(grams)

    df_ng = pd.DataFrame(counter.most_common(top_k), columns=["ngram", "frequency"])
    return df_ng


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
