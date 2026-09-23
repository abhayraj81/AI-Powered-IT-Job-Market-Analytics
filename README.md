# AI-Powered IT Job Market Analytics & Skill Gap Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)

---

## Project Overview

An end-to-end **Data Analytics + AI/NLP system** that analyses the IT job market and helps IT professionals identify skill gaps, discover required certifications, and find their best-fit specialisation — all powered by real IT role data.

---

## Problem Statement

IT professionals face three challenges:
1. **Skill visibility** — Which skills are actually demanded across IT specialisations?
2. **Career alignment** — How well does my current skillset match a target IT category?
3. **Learning prioritisation** — Which skills and certifications should I focus on first?

This project addresses all three using data analytics, NLP, and machine learning on `IT_Job_Roles_Skills.csv`.

---

## Dataset

**Source:** [Job Skill Set – Kaggle](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)

**File:** `IT_Job_Roles_Skills.csv`

| Column | Description |
|--------|-------------|
| `Job Title` | IT role title |
| `Job Description` | Full role description text |
| `Skills` | Comma-separated required skills |
| `Certifications` | Comma-separated recommended certifications |

**Key statistics after cleaning:**
- 493 raw rows → **352 unique IT roles** after deduplication
- **12 inferred IT categories** (DevOps & SRE, Software Development, Data & Analytics, Security, AI/ML, Cloud Engineering, Design & UX, etc.)
- **611 unique skills** | **416 unique certifications**
- Average **5.94 skills** and **2.50 certifications** per role

> **Note:** The dataset has no pre-assigned category column. Categories are **transparently inferred** from job titles using a keyword rule system in `src/preprocessing.py`.

---

## Features

### 💻 IT Job Market Overview
- KPI cards (roles, categories, skills, certifications)
- Category distribution, skill and certification demand charts

### 🔧 Skill Demand Analysis
- Top skills and certifications per IT category
- Normalised skill–category heatmap
- Skill co-occurrence analysis and heatmap
- Category-distinctive skill identification

### 🏢 IT Role Explorer
- Category + title drill-down
- Per-role skills, certifications, and job description
- Full category role table

### 🤖 AI / NLP Analysis
- TF-IDF category-distinctive term analysis
- Top bigrams in IT job descriptions
- ML classifier: Logistic Regression, Linear SVM, Naive Bayes
  - **Best accuracy: 72.9% (Linear SVM)** across 11 IT categories
- Live category prediction from description text
- Job similarity finder (cosine similarity)
- Skill-based role recommendation

### 🎯 IT Skill Gap Analyzer
- Select a target IT specialisation
- Enter your current skills
- Get: matched skills, missing skills, skill coverage %, learning roadmap
- **Certification recommendations** from actual dataset roles
- All recommendations 100% dataset-derived

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` + `seaborn` | Static visualisations (heatmaps, word clouds) |
| `plotly` | Interactive charts |
| `scikit-learn` | TF-IDF, ML pipelines, evaluation |
| `wordcloud` | Skill/cert word clouds |
| `nltk` | Stopword removal |
| `streamlit` | Interactive web dashboard |

---

## Project Structure

```
AI-Powered-Job-Market-Analytics/
│
├── Student_JobMarketAnalytics.ipynb   # Complete Jupyter Notebook
├── app.py                             # Streamlit dashboard (5 pages)
├── requirements.txt
├── Student_ProjectReport.docx         # Academic report (18 chapters)
├── README.md
│
├── data/
│   └── IT_Job_Roles_Skills.csv        # Dataset (place here)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Load, dedup, infer categories, normalise
│   ├── analysis.py                    # EDA, skill profiles, cert analysis
│   ├── nlp.py                         # TF-IDF, similarity, n-gram analysis
│   ├── model.py                       # ML classification pipeline
│   └── skill_gap.py                   # Gap scoring, cert recommendations
│
├── models/
│   └── it_classifier_pipeline.pkl     # Auto-generated on first run
│
└── outputs/results/                   # Exported CSVs
```

---

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK stopwords (first run only)
python -c "import nltk; nltk.download('stopwords')"

# 4. Place the dataset
# Download from Kaggle and place at:
# data/IT_Job_Roles_Skills.csv
```

---

## Running the Notebook

```bash
jupyter notebook Student_JobMarketAnalytics.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## Running the Streamlit Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Dashboard Pages:
| Page | Description |
|------|-------------|
| 💻 IT Job Market Overview | KPIs, category distribution, skill & cert charts |
| 🔧 Skill Demand Analysis | Skills per category, heatmap, co-occurrence |
| 🏢 IT Role Explorer | Role drill-down with skills, certs, descriptions |
| 🤖 AI / NLP Analysis | TF-IDF, ML classifier, job similarity |
| 🎯 Skill Gap Analyzer | Personalised gap analysis + cert recommendations |

---

## Real Results (from actual dataset analysis)

| Metric | Value |
|--------|-------|
| Unique IT roles | 352 |
| IT specialisations | 12 |
| Unique skills | 611 |
| Unique certifications | 416 |
| #1 demanded skill | Cloud Computing (82/352 roles) |
| Best ML accuracy | **72.9%** (Linear SVM, 11 classes) |
| Security F1 score | 0.86 |
| DevOps F1 score | 0.70 |

---

## Key Insights

1. **Cloud Computing is universal** — appears in 23.3% of all IT roles, confirming it as a baseline IT requirement
2. **DevOps & SRE dominates** — 24% of roles, richest skill profile (27 skills), highest certification specificity
3. **AWS certifications lead** — 4 of the top 10 certifications are AWS-specific
4. **Category signatures are distinct** — Kubernetes+Docker for DevOps; SQL+ETL for Data; CISSP for Security
5. **AI/ML is Python-first + Cloud-native** — ML and Python are the top two AI/ML role requirements
6. **72.9% ML accuracy** validates that IT categories have learnable linguistic signatures from description text alone

---

## Limitations

- 352 roles is a curated sample, not a comprehensive job market survey
- Category inference is rule-based — 7 roles (2%) remain as "Other IT"
- No seniority-level distinction (junior vs senior roles have different requirements)
- Some certification names contain PDF encoding artefacts
- IT-only scope — does not cover HR, Finance, or Sales roles

---

## Future Improvements

- Expand to 500+ roles with real-time scraping from LinkedIn/Indeed
- Add posting dates for temporal skill trend analysis
- Integrate ESCO/O*NET taxonomy for standardised skill normalisation
- Add seniority-level filtering
- Deploy on Streamlit Community Cloud

---

## Author

**Name:** Abhay Raj  
**Program:** MCA  
**Institution:** Allenhouse Institute of Technology  
**Internship:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026


