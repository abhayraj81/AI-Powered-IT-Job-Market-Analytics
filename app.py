"""
AI-Powered IT Job Market Analytics & Skill Gap Analyzer
Streamlit Interactive Dashboard - IT_Job_Roles_Skills.csv

"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    load_and_clean, get_dataset_stats, parse_user_skills, build_skill_lookup, skill_key,
)
from src.analysis import (
    category_distribution, top_job_titles, titles_by_category,
    top_skills_overall, top_skills_by_category, skills_per_category_heatmap,
    skill_cooccurrence, skill_cooccurrence_heatmap, category_unique_skills,
    generate_wordcloud, top_description_words, plot_top_words,
    skill_count_distribution,
    top_certifications_overall, top_certifications_by_category, cert_count_distribution,
)
from src.nlp import (
    build_tfidf_matrix, build_skill_tfidf, top_tfidf_terms_by_category, plot_tfidf_terms,
    extract_ngrams, plot_ngrams, find_similar_jobs, compute_similarity_matrix,
    recommend_jobs_by_skills, clean_for_nlp,
)
from src.model import (
    BASELINE_NAME, get_or_train_bundle, title_leakage_check,
    plot_confusion_matrix, plot_model_comparison,
    predict_category, predict_proba_category,
)
from src.skill_gap import (
    eligible_categories, get_category_skill_profile,
    analyze_skill_gap, plot_skill_coverage_gauge, plot_gap_breakdown,
    plot_missing_skills_priority, plot_matched_skills, format_gap_summary,
    get_category_cert_profile, plot_recommended_certs, MIN_SUPPORT, MIN_COUNT,
    MIN_ROLES_FOR_ANALYSIS,
)

# Streamlit >= 1.52 replaced use_container_width=True with width="stretch"
# (the old argument is deprecated and scheduled for removal); support both.
_ST_VERSION = tuple(int(p) for p in re.findall(r"\d+", st.__version__)[:2])
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 52) else {"use_container_width": True}

# -- Page config (must be the first Streamlit call) ---------------------------
st.set_page_config(
    page_title="IT Job Market Analytics",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1f2328;margin-bottom:.2rem}
.sub-header{font-size:1rem;color:#57606a;margin-bottom:1.5rem}
.metric-card{background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;
             padding:1rem 1.2rem;text-align:center}
.metric-value{font-size:2rem;font-weight:700;color:#3b82d4}
.metric-label{font-size:.85rem;color:#57606a}
.insight-box{background:#f0f7ff;border-left:4px solid #3b82d4;
             padding:.8rem 1rem;margin:.5rem 0;border-radius:0 6px 6px 0}
.warning-box{background:#fffbeb;border-left:4px solid #f59e0b;
             padding:.8rem 1rem;margin:.5rem 0;border-radius:0 6px 6px 0}
</style>""", unsafe_allow_html=True)

# -- Data loading (paths are relative to this file, not the working directory) -
DATA_PATH = ROOT / "data" / "IT_Job_Roles_Skills.csv"

if not DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Download it from Kaggle "
             "(Job Skill Set) and place it there.")
    st.stop()


def show_mpl(fig) -> None:
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(show_spinner="Loading IT job dataset…")
def load_data():
    return load_and_clean(str(DATA_PATH))


@st.cache_data(show_spinner="Building TF-IDF…")
def get_tfidf(_df):
    return build_tfidf_matrix(_df, text_col="job_description", max_features=3000, ngram_range=(1, 2))


@st.cache_data(show_spinner="Computing similarity matrix…")
def get_similarity(_mat):
    return compute_similarity_matrix(_mat)


@st.cache_data(show_spinner="Building skill index…")
def get_skill_index(_df):
    return build_skill_tfidf(_df)


@st.cache_resource(show_spinner="Loading IT role classifier (trains on first run)…")
def get_bundle(_df):
    return get_or_train_bundle(_df)


@st.cache_data(show_spinner="Running cross-validation (title words kept vs removed)…")
def get_leakage(_df):
    return title_leakage_check(_df)


@st.cache_data(show_spinner="Building skill profiles…")
def get_skill_profiles(_df):
    return {cat: get_category_skill_profile(_df, cat) for cat in _df["category"].unique()}


# -- Sidebar navigation --------------------------------------------------------
PAGES = [
    "💻 IT Job Market Overview",
    "🔧 Skill Demand Analysis",
    "🏢 IT Role Explorer",
    "🤖 AI / NLP Analysis",
    "🎯 Skill Gap Analyzer",
]

with st.sidebar:
    st.markdown("## 💡 Navigation")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
**Dataset**  
[IT Job Roles & Skills – Kaggle](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)

**AICTE | IBM SkillsBuild**  
Data Analytics with AI — 2026
    """)

# -- Load ----------------------------------------------------------------------
df = load_data()
stats = get_dataset_stats(df)
profiles = get_skill_profiles(df)
skill_lookup = build_skill_lookup(df)
categories = sorted(df["category"].unique().tolist())
gap_categories = eligible_categories(df)


# ==============================================================================
# PAGE 1 - IT Job Market Overview
# ==============================================================================
if page == PAGES[0]:
    st.markdown('<p class="main-header">💻 AI-Powered IT Job Market Analytics</p>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Analysis of {stats["total_jobs"]} IT roles across '
                f'{stats["total_categories"]} inferred categories.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, val) in zip(
        [c1, c2, c3, c4, c5],
        [
            ("IT Roles", stats["total_jobs"]),
            ("Categories", stats["total_categories"]),
            ("Unique Titles", stats["total_unique_titles"]),
            ("Unique Skills", stats["total_unique_skills"]),
            ("Unique Certs", stats["total_unique_certs"]),
        ],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val:,}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("IT Role Category Distribution")
    bar_fig, pie_fig, cat_counts = category_distribution(df)
    col1, col2 = st.columns(2)
    col1.plotly_chart(bar_fig, **STRETCH)
    col2.plotly_chart(pie_fig, **STRETCH)

    top = cat_counts.head(2)
    total = stats["total_jobs"]
    if len(top) == 2:
        insight = (f"<strong>Insight:</strong> {top.iloc[0]['Category']} "
                   f"({top.iloc[0]['Count'] / total:.0%}) and {top.iloc[1]['Category']} "
                   f"({top.iloc[1]['Count'] / total:.0%}) are the largest categories in this dataset. ")
    else:
        insight = ""
    st.markdown(
        f'<div class="insight-box">{insight}Categories are inferred from job titles using a '
        'transparent keyword rule system, so they reflect the rules as much as the market.</div>',
        unsafe_allow_html=True)

    st.markdown("---")

    col_sk, col_cert = st.columns(2)
    with col_sk:
        st.subheader("Most In-Demand IT Skills")
        n_skills = st.slider("Number of skills", 10, 40, 20, step=5)
        skills_fig, _ = top_skills_overall(df, n=n_skills)
        st.plotly_chart(skills_fig, **STRETCH)

    with col_cert:
        st.subheader("Most Referenced Certifications")
        n_certs = st.slider("Number of certifications", 10, 30, 15, step=5)
        cert_fig, _ = top_certifications_overall(df, n=n_certs)
        st.plotly_chart(cert_fig, **STRETCH)

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)
    col_d1.plotly_chart(skill_count_distribution(df), **STRETCH)
    col_d2.plotly_chart(cert_count_distribution(df), **STRETCH)

    st.markdown(f'<div class="insight-box">Average skills per IT role: '
                f'<strong>{stats["avg_skills_per_job"]}</strong> | '
                f'Average certifications: <strong>{stats["avg_certs_per_job"]}</strong>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Top IT Roles by Number of Required Skills")
    top_roles_fig, _ = top_job_titles(df, n=20)
    st.plotly_chart(top_roles_fig, **STRETCH)


# ==============================================================================
# PAGE 2 - Skill Demand Analysis
# ==============================================================================
elif page == PAGES[1]:
    st.markdown('<p class="main-header">🔧 IT Skill Demand Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore skill demand by IT specialisation, '
                'co-occurrence, and category skill signatures.</p>', unsafe_allow_html=True)

    selected_cat = st.selectbox("Select IT Category", ["All"] + categories)

    st.markdown("---")

    if selected_cat == "All":
        st.subheader("Top Skills — All IT Roles")
        n_skills = st.slider("Number of skills", 10, 50, 25, step=5, key="all_sk")
        fig, _ = top_skills_overall(df, n=n_skills)
        st.plotly_chart(fig, **STRETCH)

        st.subheader("Skill–Category Heatmap")
        st.markdown("*% of roles in each category requiring each skill*")
        heatmap_fig, _ = skills_per_category_heatmap(df, top_n_skills=25)
        show_mpl(heatmap_fig)

    else:
        col_sk2, col_cert2 = st.columns(2)
        with col_sk2:
            st.subheader(f"Top Skills — {selected_cat}")
            n_skills_cat = st.slider("Number of skills", 10, 40, 20, step=5, key="cat_sk")
            cat_fig, _ = top_skills_by_category(df, selected_cat, n=n_skills_cat)
            st.plotly_chart(cat_fig, **STRETCH)

        with col_cert2:
            st.subheader(f"Top Certifications — {selected_cat}")
            cert_cat_fig, _ = top_certifications_by_category(df, selected_cat, n=12)
            st.plotly_chart(cert_cat_fig, **STRETCH)

        st.subheader(f"Skill Profile for {selected_cat}")
        st.caption(f"Skills listed by ≥{MIN_SUPPORT:.0%} of roles and at least {MIN_COUNT} roles.")
        prof_view = profiles[selected_cat][["rank", "skill", "count", "frequency"]].head(30)
        if prof_view.empty:
            st.info("Not enough roles in this category to build a skill profile.")
        else:
            st.dataframe(prof_view, **STRETCH, hide_index=True)

        st.subheader(f"Skill Word Cloud — {selected_cat}")
        wc_fig = generate_wordcloud(df, kind="skills", category=selected_cat)
        if wc_fig:
            show_mpl(wc_fig)

    st.markdown("---")
    st.subheader("Skill Co-occurrence")
    top_n = st.slider("Top N skills for co-occurrence", 10, 30, 20, step=5)
    tab1, tab2 = st.tabs(["Table", "Heatmap"])
    with tab1:
        st.dataframe(skill_cooccurrence(df, top_n=top_n).head(30),
                     **STRETCH, hide_index=True)
    with tab2:
        cooccur_fig, _ = skill_cooccurrence_heatmap(df, top_n=top_n)
        st.plotly_chart(cooccur_fig, **STRETCH)

    st.markdown("---")
    st.subheader("Category-Distinctive Skills")
    st.markdown("Skills that are both **concentrated** in a category and **common** inside it "
                "(F1 of the two shares). Skills seen in fewer than 3 roles of a category are ignored.")
    unique_skills = category_unique_skills(df, top_n=6)
    cols = st.columns(3)
    for i, cat in enumerate(categories):          # ALL categories (old code showed only 9)
        with cols[i % 3]:
            st.markdown(f"**{cat}**")
            entries = unique_skills.get(cat, [])
            if not entries:
                st.caption("Not enough roles for a reliable signature.")
            for skill, score in entries[:5]:
                st.markdown(f"- {skill} `{score:.0f}`")


# ==============================================================================
# PAGE 3 - IT Role Explorer
# ==============================================================================
elif page == PAGES[2]:
    st.markdown('<p class="main-header">🏢 IT Role Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore IT roles, skills, and certifications '
                'within a selected category.</p>', unsafe_allow_html=True)

    cat3 = st.selectbox("Select IT Category", categories, key="role_cat")

    st.markdown("---")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader(f"Roles in {cat3}")
        fig_r, _ = titles_by_category(df, cat3, n=15)
        st.plotly_chart(fig_r, **STRETCH)

    with col_r2:
        st.subheader(f"Top Skills — {cat3}")
        fig_s, _ = top_skills_by_category(df, cat3, n=15)
        st.plotly_chart(fig_s, **STRETCH)

    st.markdown("---")

    st.subheader("Role Deep Dive")
    sub_df = df[df["category"] == cat3]
    all_titles = sorted(sub_df["job_title"].tolist(), key=str.lower)
    selected_title = st.selectbox("Select a job title", all_titles)
    title_row = sub_df[sub_df["job_title"] == selected_title].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Required Skills**")
        for s in title_row["skills"] or ["*None listed*"]:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**Recommended Certifications**")
        for cert in title_row["certifications"][:10] or ["*None listed*"]:
            st.markdown(f"- {cert}")

    st.markdown("**Job Description**")
    st.info(title_row["job_description"] or "No description available.")

    st.markdown("---")
    st.subheader(f"All Roles in {cat3} ({len(sub_df)} total)")
    st.dataframe(
        sub_df[["job_title", "skill_count", "cert_count", "skills_str"]].reset_index(drop=True),
        **STRETCH, hide_index=True,
    )


# ==============================================================================
# PAGE 4 - AI / NLP Analysis
# ==============================================================================
elif page == PAGES[3]:
    st.markdown('<p class="main-header">🤖 AI / NLP Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">TF-IDF insights, job similarity, '
                'ML classification, and N-gram analysis.</p>', unsafe_allow_html=True)

    tab_nlp, tab_ml, tab_sim = st.tabs(
        ["📝 TF-IDF & N-grams", "🏷️ ML Classifier", "🔍 Job Similarity"]
    )

    # -- TF-IDF & n-grams ------------------------------------------------------
    with tab_nlp:
        st.subheader("TF-IDF Category-Distinctive Terms")
        vectorizer, tfidf_matrix, _ = get_tfidf(df)
        tfidf_by_cat = top_tfidf_terms_by_category(df, vectorizer, tfidf_matrix, n=12)
        nlp_cat = st.selectbox("Select category", categories, key="nlp_cat")
        st.plotly_chart(plot_tfidf_terms(tfidf_by_cat, nlp_cat), **STRETCH)
        with st.expander("View table"):
            tdf = pd.DataFrame(tfidf_by_cat[nlp_cat], columns=["Term", "TF-IDF"])
            st.dataframe(tdf, **STRETCH, hide_index=True)

        st.markdown("---")
        st.subheader("Top Bigrams in IT Job Descriptions")
        bigrams_df = extract_ngrams(df["job_description"], n=2, top_k=30)
        st.plotly_chart(plot_ngrams(bigrams_df, "Top 30 Bigrams"), **STRETCH)

        st.markdown("---")
        st.subheader("Top Terms in Job Descriptions")
        freq_df = top_description_words(df, n=30)
        st.plotly_chart(plot_top_words(freq_df), **STRETCH)

    # -- ML classifier ---------------------------------------------------------
    with tab_ml:
        st.subheader("IT Role Category Classifier")
        bundle = get_bundle(df)
        pipeline, meta = bundle["pipeline"], bundle["meta"]
        test_df = pd.DataFrame(meta["test_results"])
        cv_df = pd.DataFrame(meta["cv_results"])
        best = test_df[test_df["Model"] == meta["selected_model"]].iloc[0]
        base = test_df[test_df["Model"] == BASELINE_NAME].iloc[0]

        title_note = ("Words that appear in the job **title** are removed from each description"
                      if meta["mask_title"] else "Title words are kept in the descriptions")
        st.markdown(f"""
The classifier predicts the **inferred category** from the **job description only**
(the skills column is not used). {title_note}, because the categories themselves come from
title keywords.

- Model chosen by **{meta['cv_folds']}-fold cross-validation** on the training split:
  **{meta['selected_model']}** (the same model is the one used for live prediction below).
- Held-out test set (**{meta['n_test']} roles**, evaluated once): accuracy **{best['Accuracy']:.1%}**,
  macro-F1 **{best['Macro F1']:.2f}** — versus **{base['Accuracy']:.1%}** accuracy for always guessing
  the largest class.
- {meta['n_classes']} categories, {meta['n_samples']} roles in total.
        """)
        st.caption("With a test set this small, a difference of a few points between models is "
                   "within noise — compare the cross-validation table too.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Cross-validation (training split)**")
            st.dataframe(cv_df.drop(columns=["Selected"], errors="ignore"),
                         **STRETCH, hide_index=True)
        with col_b:
            st.markdown("**Held-out test set**")
            st.dataframe(test_df.drop(columns=["Selected"], errors="ignore"),
                         **STRETCH, hide_index=True)

        st.plotly_chart(plot_model_comparison(test_df.drop(columns=["Selected"], errors="ignore")),
                        **STRETCH)
        st.plotly_chart(plot_confusion_matrix(meta["y_test"], meta["y_pred"], meta["classes"]),
                        **STRETCH)

        with st.expander("🔬 Circularity check: do title words leak the label?"):
            st.markdown("Categories are derived from titles. This cross-validates the same model "
                        "with title words **kept** vs **removed** from descriptions. A large drop "
                        "means the model was mostly reading the title, not the description.")
            if st.button("Run check"):
                st.dataframe(get_leakage(df), **STRETCH, hide_index=True)

        st.markdown("---")
        st.subheader("Live Category Prediction")
        user_desc = st.text_area(
            "Paste an IT job description:",
            height=120,
            placeholder="e.g. Responsible for managing Kubernetes clusters on AWS. "
                        "Experience with CI/CD, Terraform, Docker and Helm required.",
        )
        if st.button("Predict IT Category", type="primary"):
            if not user_desc.strip():
                st.warning("Please enter a job description.")
            elif not clean_for_nlp(user_desc).strip():
                st.warning("That text has no usable keywords — try a longer description.")
            else:
                pred = predict_category(pipeline, user_desc)
                proba_df = predict_proba_category(pipeline, user_desc)
                st.success(f"**Predicted Category:** {pred}")
                st.dataframe(proba_df, **STRETCH, hide_index=True)
                if not hasattr(pipeline.named_steps["clf"], "predict_proba"):
                    st.caption("Confidence is a softmax of the SVM decision scores — a relative "
                               "measure, not a calibrated probability.")

    # -- Similarity ------------------------------------------------------------
    with tab_sim:
        st.subheader("IT Job Similarity & Recommendation")
        sim_tab1, sim_tab2 = st.tabs(["Find Similar Roles", "Skill-Based Role Search"])

        with sim_tab1:
            v, mat, _ = get_tfidf(df)
            sim_matrix = get_similarity(mat)
            filter_cat = st.selectbox("Filter by category", ["All"] + categories, key="sim_cat")
            role_idx = (df.index[df["category"] == filter_cat].tolist()
                        if filter_cat != "All" else df.index.tolist())
            role_idx = sorted(role_idx, key=lambda i: df.loc[i, "job_title"].lower())
            # every role is selectable (the old list was cut off after 150 entries)
            chosen_idx = st.selectbox("Select a job role", role_idx,
                                      format_func=lambda i: df.loc[i, "job_title"], key="sim_role")
            n_sim = st.slider("Number of similar roles", 3, 15, 5, key="n_sim")

            if st.button("Find Similar Roles"):
                sim_jobs = find_similar_jobs(df, sim_matrix, chosen_idx, top_n=n_sim)
                st.dataframe(sim_jobs, **STRETCH, hide_index=True)
                sim_bar = px.bar(
                    sim_jobs, x="similarity_score", y="job_title", orientation="h",
                    color="similarity_score", color_continuous_scale="Blues",
                    title="Similarity Scores",
                )
                sim_bar.update_layout(yaxis={"categoryorder": "total ascending"},
                                      coloraxis_showscale=False)
                st.plotly_chart(sim_bar, **STRETCH)

        with sim_tab2:
            skill_input = st.text_input(
                "Your skills (comma-separated):",
                placeholder="Python, AWS, Docker, Kubernetes, CI/CD",
                key="sim_skills",
            )
            n_rec = st.slider("Recommendations", 5, 20, 10, key="n_rec")
            if st.button("Find Matching Roles", type="primary"):
                user_list = parse_user_skills(skill_input, lookup=skill_lookup)
                if not user_list:
                    st.warning("Enter at least one skill.")
                else:
                    unknown = [s for s in user_list if skill_key(s) not in skill_lookup]
                    if unknown:
                        st.caption("Not found in the dataset's skill vocabulary (check spelling): "
                                   + ", ".join(unknown))
                    sk_vec, sk_mat = get_skill_index(df)
                    recs = recommend_jobs_by_skills(df, user_list, sk_vec, sk_mat, top_n=n_rec)
                    if recs.empty:
                        st.warning("None of these skills appear in any role in the dataset.")
                    else:
                        st.dataframe(recs, **STRETCH, hide_index=True)


# ==============================================================================
# PAGE 5 - Skill Gap Analyzer
# ==============================================================================
elif page == PAGES[4]:
    st.markdown('<p class="main-header">🎯 IT Skill Gap Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Find out which IT skills and certifications you need for '
        'your target specialisation.</p>', unsafe_allow_html=True)

    st.subheader("Step 1: Select Your Target IT Specialisation")
    target_cat = st.selectbox("Target Category", options=gap_categories, key="gap_cat")
    hidden = [c for c in categories if c not in gap_categories]
    if hidden:
        st.caption(f"Not offered (fewer than {MIN_ROLES_FOR_ANALYSIS} roles, so profiles would be "
                   f"unreliable): {', '.join(hidden)}.")
    profile = profiles[target_cat]

    if profile.empty:
        st.error(f"No skill profile found for {target_cat}.")
        st.stop()

    role_count = int((df["category"] == target_cat).sum())
    st.info(
        f"📋 Skill profile for **{target_cat}**: **{len(profile)} skills** required by at least "
        f"{MIN_SUPPORT:.0%} of the **{role_count}** roles in this category."
    )

    with st.expander("View full skill profile"):
        st.dataframe(profile[["rank", "skill", "count", "frequency"]],
                     **STRETCH, hide_index=True)

    st.subheader("Step 2: Enter Your Current Skills")
    st.markdown("Separate skills with commas — capitalisation does not matter. "
                "Example: `python, aws, docker, ci/cd, linux`")
    user_input = st.text_area(
        "Your current skills:",
        height=80,
        placeholder="Python, AWS, Docker, Kubernetes, Linux, Git",
        key="user_skills",
    )

    if st.button("Analyze My IT Skill Gap", type="primary", **STRETCH):
        user_skills = parse_user_skills(user_input, lookup=skill_lookup)
        if not user_skills:
            st.warning("Please enter at least one skill.")
        else:
            unknown = [s for s in user_skills if skill_key(s) not in skill_lookup]
            if unknown:
                st.caption("Not found anywhere in the dataset (check spelling): " + ", ".join(unknown))

            result = analyze_skill_gap(user_skills, profile)
            if result.get("error"):
                st.error(result["error"])
                st.stop()

            st.markdown("---")
            st.subheader("📊 Your IT Skill Gap Results")

            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            kc1.metric("Weighted coverage", f"{result['coverage_pct']}%")
            kc2.metric("Simple coverage", f"{result['simple_coverage_pct']}%")
            kc3.metric("Matched Skills", len(result["matched_skills"]))
            kc4.metric("Missing Skills", len(result["missing_skills"]))
            kc5.metric("Extra Skills", len(result["extra_skills"]))
            st.caption("Weighted coverage counts each skill by how many roles need it, so common "
                       "skills matter more than niche ones. Simple coverage counts them equally.")

            st.markdown("---")
            col_g, col_b = st.columns(2)
            col_g.plotly_chart(plot_skill_coverage_gauge(result["coverage_pct"]),
                               **STRETCH)
            col_b.plotly_chart(plot_gap_breakdown(result["matched_skills"],
                                                  result["missing_skills"]),
                               **STRETCH)

            st.markdown("---")
            if result["missing_skills"]:
                st.subheader("🚀 Skills to Learn (Priority Order)")
                st.plotly_chart(
                    plot_missing_skills_priority(profile, result["missing_skills"], top_n=15),
                    **STRETCH,
                )

            if result["matched_skills"]:
                st.subheader("✅ Your Matched Skills")
                st.plotly_chart(
                    plot_matched_skills(profile, result["matched_skills"]),
                    **STRETCH,
                )

            st.markdown("---")
            st.subheader("📜 Recommended Certifications for This Role")
            cert_profile = get_category_cert_profile(df, target_cat, top_n=8)
            if not cert_profile.empty:
                col_cert_chart, col_cert_list = st.columns(2)
                col_cert_chart.plotly_chart(
                    plot_recommended_certs(cert_profile, top_n=8),
                    **STRETCH,
                )
                col_cert_list.markdown("**Top Certifications:**")
                for _, row in cert_profile.iterrows():
                    col_cert_list.markdown(f"- {row['certification']} `{row['count']}`")

            st.markdown("---")
            col_m, col_miss = st.columns(2)
            with col_m:
                st.markdown("### ✅ Skills You Have")
                for s in result["matched_skills"] or ["*None matched*"]:
                    st.markdown(f"- {s}")

            with col_miss:
                st.markdown("### ❌ Skills to Acquire")
                if result["priority_missing"]:
                    for s in result["priority_missing"]:
                        row = profile[profile["skill"] == s]
                        freq = f"{row['frequency'].values[0]:.1f}%" if not row.empty else ""
                        st.markdown(f"- {s} *(in {freq} of roles)*")
                else:
                    st.success("🎉 You already have all key skills for this role!")

            if result["extra_skills"]:
                with st.expander("➕ Your Additional Skills (Beyond Core Profile)"):
                    for s in result["extra_skills"]:
                        st.markdown(f"- {s}")

            with st.expander("📄 Text Summary"):
                st.markdown(format_gap_summary(result, target_cat))

            st.markdown(
                f'<div class="warning-box"><strong>Note:</strong> Recommendations reflect patterns '
                f'in {stats["total_jobs"]} IT role definitions and serve as a learning guide, '
                f'not a guarantee of employment.</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ How the IT Skill Gap Analyzer Works"):
        st.markdown(f"""
**Methodology:**

1. **Role skill profile:** skills listed by ≥{MIN_SUPPORT:.0%} of the roles in the target category
   (and by at least {MIN_COUNT} roles) form the required profile. Categories with fewer than
   {MIN_ROLES_FOR_ANALYSIS} roles are not offered.

2. **Normalisation:** your input is matched case-insensitively and mapped to the dataset's spelling
   (e.g. `k8s → Kubernetes`, `ci/cd → CI/CD`, `python → Python`).

3. **Gap calculation:**
   - `Matched = your skills ∩ profile`
   - `Simple coverage = |Matched| / |Profile| × 100`
   - `Weighted coverage = Σ demand(matched) / Σ demand(profile) × 100`,
     where demand = % of roles in the category that list the skill
   - Missing skills are sorted by demand → your learning roadmap.

4. **Certifications:** the most frequent certifications among the dataset's roles in that category.

**Limitations:** small dataset, no seniority levels, categories inferred from titles.
        """)
