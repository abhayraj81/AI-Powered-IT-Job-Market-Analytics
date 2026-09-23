"""
app.py
======
AI-Powered IT Job Market Analytics & Skill Gap Analyzer
Streamlit Interactive Dashboard — IT_Job_Roles_Skills.csv

Run:  streamlit run app.py
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import load_and_clean, get_dataset_stats, parse_user_skills
from src.analysis import (
    category_distribution, top_job_titles, titles_by_category,
    top_skills_overall, top_skills_by_category, skills_per_category_heatmap,
    skill_cooccurrence, skill_cooccurrence_heatmap, category_unique_skills,
    generate_wordcloud, top_description_words, plot_top_words,
    skill_count_distribution, get_skill_counts,
    top_certifications_overall, top_certifications_by_category, cert_count_distribution,
)
from src.nlp import (
    build_tfidf_matrix, top_tfidf_terms_by_category, plot_tfidf_terms,
    extract_ngrams, plot_ngrams, find_similar_jobs, compute_similarity_matrix,
    recommend_jobs_by_skills,
)
from src.model import (
    prepare_features, split_data, train_and_evaluate_all, train_best_model,
    plot_confusion_matrix, plot_model_comparison, save_model, load_model,
    predict_category, predict_proba_category,
)
from src.skill_gap import (
    get_category_skill_profile, parse_user_skills as sg_parse_skills,
    analyze_skill_gap, plot_skill_coverage_gauge, plot_gap_breakdown,
    plot_missing_skills_priority, plot_matched_skills, format_gap_summary,
    get_category_cert_profile, plot_recommended_certs,
)

# ── Page config ──────────────────────────────────────────────────────────────
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

# ── Data loading (cached) ────────────────────────────────────────────────────
DATA_PATH = "data/IT_Job_Roles_Skills.csv"


@st.cache_data(show_spinner="Loading IT job dataset…")
def load_data():
    return load_and_clean(DATA_PATH)


@st.cache_data(show_spinner="Building TF-IDF…")
def get_tfidf(_df):
    return build_tfidf_matrix(_df, text_col="job_description",
                               max_features=3000, ngram_range=(1, 2))


@st.cache_data(show_spinner="Computing similarity matrix…")
def get_similarity(_mat):
    return compute_similarity_matrix(_mat)


@st.cache_data(show_spinner="Building skill TF-IDF…")
def get_skill_tfidf(_df):
    return build_tfidf_matrix(_df, text_col="skills_str",
                               max_features=2000, ngram_range=(1, 1))


@st.cache_resource(show_spinner="Training IT role classifier…")
def get_trained_model(_df):
    pipeline = load_model()
    if pipeline is not None:
        return pipeline, None, None, None, None
    X, y, _ = prepare_features(_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20)
    results_df = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    pipeline, y_pred, report, cats = train_best_model(
        X_train, X_test, y_train, y_test, best_name="Logistic Regression"
    )
    save_model(pipeline)
    return pipeline, y_pred, y_test, results_df, cats


@st.cache_data(show_spinner="Building skill profiles…")
def get_skill_profiles(_df):
    return {cat: get_category_skill_profile(_df, cat)
            for cat in _df["category"].unique()}


# ── Sidebar navigation ───────────────────────────────────────────────────────
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

# ── Load ────────────────────────────────────────────────────────────────────
df = load_data()
stats = get_dataset_stats(df)
profiles = get_skill_profiles(df)
categories = sorted(df["category"].unique().tolist())


# ============================================================================
# PAGE 1 — IT Job Market Overview
# ============================================================================
if page == PAGES[0]:
    st.markdown('<p class="main-header">💻 AI-Powered IT Job Market Analytics</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of 352 IT roles across '
                '12 specialisation categories.</p>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, val) in zip(
        [c1, c2, c3, c4, c5],
        [
            ("IT Roles", stats["total_jobs"]),
            ("Specialisations", stats["total_categories"]),
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
    col1.plotly_chart(bar_fig, use_container_width=True)
    col2.plotly_chart(pie_fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box"><strong>Insight:</strong> DevOps & SRE (24%) and Software '
        'Development (19%) dominate the IT role dataset — reflecting the high market demand '
        'for these profiles. Categories are inferred from job titles using a transparent '
        'keyword rule system.</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_sk, col_cert = st.columns(2)
    with col_sk:
        st.subheader("Most In-Demand IT Skills")
        n_skills = st.slider("Number of skills", 10, 40, 20, step=5)
        skills_fig, _ = top_skills_overall(df, n=n_skills)
        st.plotly_chart(skills_fig, use_container_width=True)

    with col_cert:
        st.subheader("Most Referenced Certifications")
        n_certs = st.slider("Number of certifications", 10, 30, 15, step=5)
        cert_fig, _ = top_certifications_overall(df, n=n_certs)
        st.plotly_chart(cert_fig, use_container_width=True)

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)
    col_d1.plotly_chart(skill_count_distribution(df), use_container_width=True)
    col_d2.plotly_chart(cert_count_distribution(df), use_container_width=True)

    st.markdown(f'<div class="insight-box">Average skills per IT role: '
                f'<strong>{stats["avg_skills_per_job"]}</strong> | '
                f'Average certifications: <strong>{stats["avg_certs_per_job"]}</strong>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Top IT Roles by Number of Required Skills")
    top_roles_fig, _ = top_job_titles(df, n=20)
    st.plotly_chart(top_roles_fig, use_container_width=True)


# ============================================================================
# PAGE 2 — Skill Demand Analysis
# ============================================================================
elif page == PAGES[1]:
    st.markdown('<p class="main-header">🔧 IT Skill Demand Analysis</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore skill demand by IT specialisation, '
                'co-occurrence, and category skill signatures.</p>', unsafe_allow_html=True)

    selected_cat = st.selectbox("Select IT Category", ["All"] + categories)

    st.markdown("---")

    if selected_cat == "All":
        st.subheader("Top Skills — All IT Roles")
        n_skills = st.slider("Number of skills", 10, 50, 25, step=5, key="all_sk")
        fig, _ = top_skills_overall(df, n=n_skills)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Skill–Category Heatmap")
        st.markdown("*% of roles in each category requiring each skill*")
        heatmap_fig, _ = skills_per_category_heatmap(df, top_n_skills=25)
        st.pyplot(heatmap_fig)

    else:
        col_sk2, col_cert2 = st.columns(2)
        with col_sk2:
            st.subheader(f"Top Skills — {selected_cat}")
            n_skills_cat = st.slider("Number of skills", 10, 40, 20, step=5, key="cat_sk")
            cat_fig, _ = top_skills_by_category(df, selected_cat, n=n_skills_cat)
            st.plotly_chart(cat_fig, use_container_width=True)

        with col_cert2:
            st.subheader(f"Top Certifications — {selected_cat}")
            cert_cat_fig, _ = top_certifications_by_category(df, selected_cat, n=12)
            st.plotly_chart(cert_cat_fig, use_container_width=True)

        # Skill profile table
        st.subheader(f"Skill Profile for {selected_cat}")
        st.dataframe(profiles[selected_cat].head(30), use_container_width=True, hide_index=True)

        # Word cloud
        st.subheader(f"Skill Word Cloud — {selected_cat}")
        wc_fig = generate_wordcloud(df, column="skills_str", category=selected_cat)
        if wc_fig:
            st.pyplot(wc_fig)

    st.markdown("---")
    st.subheader("Skill Co-occurrence")
    top_n = st.slider("Top N skills for co-occurrence", 10, 30, 20, step=5)
    tab1, tab2 = st.tabs(["Table", "Heatmap"])
    with tab1:
        st.dataframe(skill_cooccurrence(df, top_n=top_n).head(30),
                     use_container_width=True, hide_index=True)
    with tab2:
        cooccur_fig, _ = skill_cooccurrence_heatmap(df, top_n=top_n)
        st.plotly_chart(cooccur_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Category-Distinctive Skills")
    st.markdown("Skills with the highest concentration in one category vs the global average.")
    unique_skills = category_unique_skills(df, top_n=6)
    cols = st.columns(3)
    for i, cat in enumerate(sorted(categories)[:9]):
        with cols[i % 3]:
            st.markdown(f"**{cat}**")
            if cat in unique_skills:
                for skill, score in unique_skills[cat][:5]:
                    st.markdown(f"- {skill} `{score:.0f}%`")


# ============================================================================
# PAGE 3 — IT Role Explorer
# ============================================================================
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
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r2:
        st.subheader(f"Top Skills — {cat3}")
        fig_s, _ = top_skills_by_category(df, cat3, n=15)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # Title drill-down
    st.subheader("Role Deep Dive")
    sub_df = df[df["category"] == cat3]
    all_titles = sorted(sub_df["job_title"].tolist())
    selected_title = st.selectbox("Select a job title", all_titles)
    title_row = sub_df[sub_df["job_title"] == selected_title].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Required Skills**")
        for s in title_row["skills"]:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**Recommended Certifications**")
        for cert in title_row["certifications"][:10]:
            st.markdown(f"- {cert}")

    st.markdown("**Job Description**")
    st.info(title_row["job_description"])

    st.markdown("---")
    st.subheader(f"All Roles in {cat3} ({len(sub_df)} total)")
    st.dataframe(
        sub_df[["job_title", "skill_count", "cert_count", "skills_str"]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )


# ============================================================================
# PAGE 4 — AI / NLP Analysis
# ============================================================================
elif page == PAGES[3]:
    st.markdown('<p class="main-header">🤖 AI / NLP Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">TF-IDF insights, job similarity, '
                'ML classification, and N-gram analysis.</p>', unsafe_allow_html=True)

    tab_nlp, tab_ml, tab_sim = st.tabs(
        ["📝 TF-IDF & N-grams", "🏷️ ML Classifier", "🔍 Job Similarity"]
    )

    with tab_nlp:
        st.subheader("TF-IDF Category-Distinctive Terms")
        vectorizer, tfidf_matrix, _ = get_tfidf(df)
        tfidf_by_cat = top_tfidf_terms_by_category(df, vectorizer, tfidf_matrix, n=12)
        nlp_cat = st.selectbox("Select category", categories, key="nlp_cat")
        st.plotly_chart(plot_tfidf_terms(tfidf_by_cat, nlp_cat), use_container_width=True)
        with st.expander("View table"):
            tdf = pd.DataFrame(tfidf_by_cat[nlp_cat], columns=["Term", "TF-IDF"])
            st.dataframe(tdf, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Top Bigrams in IT Job Descriptions")
        bigrams_df = extract_ngrams(df["job_description"], n=2, top_k=30)
        st.plotly_chart(plot_ngrams(bigrams_df, "Top 30 Bigrams"), use_container_width=True)

        st.markdown("---")
        st.subheader("Top Terms in Job Descriptions")
        freq_df = top_description_words(df, n=30)
        st.plotly_chart(plot_top_words(freq_df), use_container_width=True)

    with tab_ml:
        st.subheader("IT Role Category Classifier")
        st.markdown("""
A **Logistic Regression** model trained on TF-IDF features from job descriptions predicts
the IT role category. The **skills column is not used** to prevent data leakage.

> With 352 samples across 11 categories, the model achieves **~61% accuracy**
> (Logistic Regression). Linear SVM achieves **72.9%** — the best model is saved.
        """)

        with st.spinner("Loading classifier…"):
            pipeline, y_pred, y_test, results_df, ml_cats = get_trained_model(df)

        if results_df is not None:
            st.plotly_chart(plot_model_comparison(results_df), use_container_width=True)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.plotly_chart(plot_confusion_matrix(y_test, y_pred, ml_cats),
                            use_container_width=True)

        st.markdown("---")
        st.subheader("Live Category Prediction")
        user_desc = st.text_area(
            "Paste an IT job description:",
            height=120,
            placeholder="e.g. Responsible for managing Kubernetes clusters on AWS. "
                        "Experience with CI/CD, Terraform, Docker and Helm required.",
        )
        if st.button("Predict IT Category", type="primary"):
            if user_desc.strip():
                pred = predict_category(pipeline, user_desc)
                proba_df = predict_proba_category(pipeline, user_desc)
                st.success(f"**Predicted Category:** {pred}")
                st.dataframe(proba_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Please enter a job description.")

    with tab_sim:
        st.subheader("IT Job Similarity & Recommendation")
        sim_tab1, sim_tab2 = st.tabs(["Find Similar Roles", "Skill-Based Role Search"])

        with sim_tab1:
            v, mat, _ = get_tfidf(df)
            sim_matrix = get_similarity(mat)
            filter_cat = st.selectbox("Filter by category", ["All"] + categories, key="sim_cat")
            if filter_cat != "All":
                sub_idx = df[df["category"] == filter_cat].index.tolist()
                opts = {f"{i}: {df.loc[i,'job_title']}": i for i in sub_idx}
            else:
                opts = {f"{i}: {df.loc[i,'job_title']}": i for i in df.index}
            chosen = st.selectbox("Select a job role", list(opts.keys())[:150], key="sim_role")
            chosen_idx = opts[chosen]
            n_sim = st.slider("Number of similar roles", 3, 15, 5, key="n_sim")

            if st.button("Find Similar Roles"):
                sim_jobs = find_similar_jobs(df, sim_matrix, chosen_idx, top_n=n_sim)
                st.dataframe(sim_jobs, use_container_width=True, hide_index=True)
                sim_bar = px.bar(
                    sim_jobs, x="similarity_score", y="job_title", orientation="h",
                    color="similarity_score", color_continuous_scale="Blues",
                    title="Similarity Scores",
                )
                sim_bar.update_layout(yaxis={"categoryorder": "total ascending"},
                                       coloraxis_showscale=False)
                st.plotly_chart(sim_bar, use_container_width=True)

        with sim_tab2:
            skill_input = st.text_input(
                "Your skills (comma-separated):",
                placeholder="Python, AWS, Docker, Kubernetes, CI/CD",
                key="sim_skills",
            )
            n_rec = st.slider("Recommendations", 5, 20, 10, key="n_rec")
            if st.button("Find Matching Roles", type="primary"):
                if skill_input.strip():
                    user_list = parse_user_skills(skill_input)
                    sk_vec, sk_tfidf, _ = get_skill_tfidf(df)
                    recs = recommend_jobs_by_skills(df, user_list, sk_vec, sk_tfidf, top_n=n_rec)
                    st.dataframe(recs, use_container_width=True, hide_index=True)
                else:
                    st.warning("Enter at least one skill.")


# ============================================================================
# PAGE 5 — Skill Gap Analyzer
# ============================================================================
elif page == PAGES[4]:
    st.markdown('<p class="main-header">🎯 IT Skill Gap Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Find out which IT skills and certifications you need for '
        'your target specialisation.</p>', unsafe_allow_html=True)

    # ── Step 1: Category ─────────────────────────────────────────────────────
    st.subheader("Step 1: Select Your Target IT Specialisation")
    target_cat = st.selectbox("Target Category", options=categories, key="gap_cat")
    profile = profiles[target_cat]

    if profile.empty:
        st.error(f"No skill profile found for {target_cat}.")
        st.stop()

    role_count = df[df["category"] == target_cat].shape[0]
    st.info(
        f"📋 Skill profile for **{target_cat}** built from **{len(profile)} skills** "
        f"across **{role_count}** IT role definitions."
    )

    with st.expander("View full skill profile"):
        st.dataframe(profile[["rank", "skill", "count", "frequency"]], use_container_width=True,
                     hide_index=True)

    # ── Step 2: User skills ──────────────────────────────────────────────────
    st.subheader("Step 2: Enter Your Current Skills")
    st.markdown("Separate skills with commas. Example: `Python, AWS, Docker, CI/CD, Linux`")
    user_input = st.text_area(
        "Your current skills:",
        height=80,
        placeholder="Python, AWS, Docker, Kubernetes, Linux, Git",
        key="user_skills",
    )

    # ── Analyze ──────────────────────────────────────────────────────────────
    if st.button("Analyze My IT Skill Gap", type="primary", use_container_width=True):
        user_skills = sg_parse_skills(user_input)
        if not user_skills:
            st.warning("Please enter at least one skill.")
        else:
            result = analyze_skill_gap(user_skills, profile)

            if result.get("error"):
                st.error(result["error"])
                st.stop()

            st.markdown("---")
            st.subheader("📊 Your IT Skill Gap Results")

            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Coverage", f"{result['coverage_pct']}%")
            kc2.metric("Matched Skills", len(result["matched_skills"]))
            kc3.metric("Missing Skills", len(result["missing_skills"]))
            kc4.metric("Extra Skills", len(result["extra_skills"]))

            st.markdown("---")
            col_g, col_b = st.columns(2)
            col_g.plotly_chart(plot_skill_coverage_gauge(result["coverage_pct"]),
                               use_container_width=True)
            col_b.plotly_chart(plot_gap_breakdown(result["matched_skills"],
                                                   result["missing_skills"]),
                               use_container_width=True)

            st.markdown("---")
            if result["missing_skills"]:
                st.subheader("🚀 Skills to Learn (Priority Order)")
                st.plotly_chart(
                    plot_missing_skills_priority(profile, result["missing_skills"], top_n=15),
                    use_container_width=True,
                )

            if result["matched_skills"]:
                st.subheader("✅ Your Matched Skills")
                st.plotly_chart(
                    plot_matched_skills(profile, result["matched_skills"]),
                    use_container_width=True,
                )

            # Certifications
            st.markdown("---")
            st.subheader("📜 Recommended Certifications for This Role")
            cert_profile = get_category_cert_profile(df, target_cat, top_n=8)
            if not cert_profile.empty:
                col_cert_chart, col_cert_list = st.columns(2)
                col_cert_chart.plotly_chart(
                    plot_recommended_certs(cert_profile, top_n=8),
                    use_container_width=True,
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
                '<div class="warning-box"><strong>Note:</strong> Skill recommendations are '
                'based on patterns in 352 IT role definitions and serve as a learning guide, '
                'not a guarantee of employment.</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ How the IT Skill Gap Analyzer Works"):
        st.markdown("""
**Methodology:**

1. **Role Skill Profile:** Skills appearing in ≥8% of roles in the target IT category form the required profile.

2. **Skill Normalisation:** Your input is normalised (e.g. `k8s → Kubernetes`, `ci/cd → CI/CD`).

3. **Gap Calculation:**
   - `Matched = User Skills ∩ Required Profile`
   - `Coverage (%) = |Matched| / |Required| × 100`
   - Missing skills are sorted by demand frequency → your learning roadmap.

4. **Certification Recommendations:** Top certifications from actual dataset roles in the target category.

**Limitations:** Based on 352 IT role definitions. Does not distinguish seniority levels.
        """)
