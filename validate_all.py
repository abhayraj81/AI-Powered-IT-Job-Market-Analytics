"""
Project validation script.

Run from anywhere:   python validate_all.py
Exit code 0 = every check passed, 1 = at least one check failed.

Unlike the old version (which printed "ALL VALIDATIONS PASSED" no matter what,
except for missing files), every step here asserts something real:
  * preprocessing rules and data-quality invariants
  * analysis / NLP outputs are non-empty and consistent
  * the classifier beats a majority-class baseline, the SAVED model is the one
    reported as best, and predictions work for every model type
  * the skill-gap analysis is case-insensitive and bounded
The model is trained into a temporary folder, so validating does not overwrite
models/it_classifier_pipeline.pkl (the app trains/loads that file itself).
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

failures: list = []
warnings_seen: list = []


def check(condition: bool, message: str) -> None:
    print(("[OK]   " if condition else "[FAIL] ") + message)
    if not condition:
        failures.append(message)


def warn_if(condition: bool, message: str) -> None:
    if condition:
        print("[WARN] " + message)
        warnings_seen.append(message)


print("=== VALIDATION (IT_Job_Roles_Skills.csv) ===")

# 1. Preprocessing

print("\n-- 1. Preprocessing")
from src.preprocessing import (
    load_and_clean, get_dataset_stats, infer_category, skill_key,
    build_skill_lookup, parse_user_skills,
)

DATA_PATH = ROOT / "data" / "IT_Job_Roles_Skills.csv"
check(DATA_PATH.exists(), f"dataset present at {DATA_PATH.relative_to(ROOT)}")
if not DATA_PATH.exists():
    sys.exit(1)

df = load_and_clean(str(DATA_PATH))
stats = get_dataset_stats(df)
print(f"Shape: {df.shape} | categories: {df['category'].value_counts().to_dict()}")

expected_cols = {"job_title", "job_description", "skills", "certifications", "category",
                 "skill_count", "cert_count", "skills_str", "certs_str"}
check(expected_cols <= set(df.columns), "all expected columns present")
check(len(df) > 0 and (df["job_title"].str.strip() != "").all(), "no empty job titles")
check(df["job_title"].str.upper().is_unique, "job titles are unique (case-insensitive)")
check(not any(skill_key(s) == "nan" for sk in df["skills"] for s in sk),
      'missing values did not become a fake "nan" skill')
check(all(len({skill_key(s) for s in sk}) == len(sk) for sk in df["skills"]),
      "no duplicate skills inside a role (case-insensitive)")
forms: dict = {}
for sk in df["skills"]:
    for s in sk:
        forms.setdefault(skill_key(s), set()).add(s)
check(all(len(v) == 1 for v in forms.values()), "one spelling per skill across the dataset")
check((df["skill_count"] == df["skills"].apply(len)).all(), "skill_count matches skills")

# rule regressions (word-boundary matching)
check(infer_category("Linux Administrator") == "Networking & Infrastructure",
      '"Linux Administrator" is not mistaken for UX')
check(infer_category("Vector Specialist") == "Other IT", '"cto" is not matched inside "Vector"')
check(infer_category("UX Designer") == "Design & UX", "real UX titles still map to Design & UX")
warn_if(stats["category_counts"].get("Other IT", 0) / stats["total_jobs"] > 0.25,
        "more than 25% of roles fall into 'Other IT' - the category rules may need extending")
print(f"[info] {stats['total_jobs']} roles, {stats['total_categories']} categories, "
      f"{stats['total_unique_skills']} unique skills, {stats['total_unique_certs']} unique certs")

# ---------------------------------------------------------------------------
# 2. Analysis
# ---------------------------------------------------------------------------
print("\n-- 2. Analysis")
from src.analysis import (
    category_distribution, top_skills_overall, skill_cooccurrence,
    skills_per_category_heatmap, top_certifications_overall, generate_wordcloud,
    category_unique_skills, get_skill_counts, top_description_words,
)

_, _, counts = category_distribution(df)
check(int(counts["Count"].sum()) == len(df), "category counts add up to the number of roles")
_, sk_df = top_skills_overall(df, n=15)
check(len(sk_df) > 0, "top skills computed")
_, cert_df = top_certifications_overall(df, n=10)
check(len(cert_df) > 0, "top certifications computed")
check(not skill_cooccurrence(df, top_n=20).empty, "skill co-occurrence computed")
fig, heat = skills_per_category_heatmap(df, top_n_skills=20)
plt.close(fig)
check(heat.shape[0] == df["category"].nunique(), "heatmap has one row per category")
wc = generate_wordcloud(df, kind="skills")
check(wc is not None, "skill word cloud built")
plt.close("all")
check(set(category_unique_skills(df)) == set(df["category"].unique()),
      "distinctive skills computed for every category")
check(not top_description_words(df, n=10).empty, "description word frequencies computed")
print("Top 5 skills:", sk_df.head(5)["Skill"].tolist())
print("Top 5 certs: ", cert_df.head(5)["Certification"].tolist())

# ---------------------------------------------------------------------------
# 3. NLP
# ---------------------------------------------------------------------------
print("\n-- 3. NLP")
from src.nlp import (
    build_tfidf_matrix, top_tfidf_terms_by_category, extract_ngrams,
    compute_similarity_matrix, tokenize, build_skill_tfidf, recommend_jobs_by_skills,
)

vec, mat, _ = build_tfidf_matrix(df, text_col="job_description", max_features=3000)
check(mat.shape[0] == len(df) and mat.shape[1] > 0, f"TF-IDF matrix {mat.shape}")
sim = compute_similarity_matrix(mat)
has_terms = mat.getnnz(axis=1) > 0
check(sim.shape == (len(df), len(df)) and np.allclose(np.diag(sim)[has_terms], 1.0),
      "similarity matrix is square with self-similarity 1")
check(set(top_tfidf_terms_by_category(df, vec, mat, n=5)) == set(df["category"].unique()),
      "TF-IDF terms computed for every category")
check(not extract_ngrams(df["job_description"], n=2, top_k=20).empty, "bigrams extracted")
check({"csharp", "cpp", "r", "cicd", "s3"} <= set(tokenize("C#, C++, R, CI/CD and S3")),
      "tokeniser keeps C#, C++, R, CI/CD and S3")
sk_vec, sk_mat = build_skill_tfidf(df)
top_skill = get_skill_counts(df).index[0]
r_low = recommend_jobs_by_skills(df, [top_skill.lower()], sk_vec, sk_mat)
r_up = recommend_jobs_by_skills(df, [top_skill.upper()], sk_vec, sk_mat)
check(not r_low.empty and r_low.equals(r_up), "skill-based recommendations are case-insensitive")

# ---------------------------------------------------------------------------
# 4. ML
# ---------------------------------------------------------------------------
print("\n-- 4. ML classifier (trained in a temp folder)")
from src import model as M

X, y, valid_cats = M.prepare_features(df)
print(f"ML-eligible categories ({len(valid_cats)}): {valid_cats}")
check(len(valid_cats) >= 2, "at least two categories have enough roles to train on")

with tempfile.TemporaryDirectory() as tmp:
    model_path = Path(tmp) / "model.pkl"
    bundle = M.train_and_save(df, model_path)
    reloaded = M.load_bundle(model_path)

meta, pipe = bundle["meta"], bundle["pipeline"]
cv_tbl, test_tbl = pd.DataFrame(meta["cv_results"]), pd.DataFrame(meta["test_results"])
print(f"\nCross-validation ({meta['cv_folds']}-fold, training split only):")
print(cv_tbl.drop(columns=["Selected"], errors="ignore").to_string(index=False))
print(f"\nHeld-out test set ({meta['n_test']} roles, scored once):")
print(test_tbl.drop(columns=["Selected"], errors="ignore").to_string(index=False))

selected = meta["selected_model"]
sel = test_tbl[test_tbl["Model"] == selected].iloc[0]
base = test_tbl[test_tbl["Model"] == M.BASELINE_NAME].iloc[0]
served_type = type(pipe.named_steps["clf"]).__name__
type_by_name = {"Logistic Regression": "LogisticRegression", "Linear SVM": "LinearSVC",
                "Naive Bayes": "MultinomialNB"}

check(reloaded is not None and reloaded["meta"]["test_results"] == meta["test_results"],
      "saved model reloads together with its reported results")
check(type_by_name[selected] == served_type,
      f"the SAVED model ({served_type}) is the one reported as best ({selected})")
check(sel["Accuracy"] > base["Accuracy"] and sel["Macro F1"] > base["Macro F1"],
      f"{selected} beats the majority-class baseline "
      f"(acc {sel['Accuracy']:.1%} vs {base['Accuracy']:.1%}, "
      f"macro-F1 {sel['Macro F1']:.2f} vs {base['Macro F1']:.2f})")
pred = M.predict_category(pipe, "Responsible for deploying and managing Kubernetes clusters on AWS")
check(pred in meta["classes"], f"live prediction returns a known category ({pred})")
proba = M.predict_proba_category(pipe, "Responsible for deploying and managing Kubernetes clusters on AWS")
check(abs(proba["Confidence (%)"].sum() - 100) < 1.0, "confidence scores sum to 100%")
for name, clf in M.get_classifiers().items():   # every model type must support prediction output
    p = M.build_pipeline(clf).fit(X, y)
    out = M.predict_proba_category(p, "network firewall security audit")
    check(abs(out["Confidence (%)"].sum() - 100) < 1.0, f"confidence output works for {name}")

leak = M.title_leakage_check(df)
print("\nTitle-leakage check (CV, Logistic Regression):")
print(leak.to_string(index=False))
kept = leak.loc[leak["Title words in descriptions"] == "kept", "CV Macro F1"].iloc[0]
removed = leak.loc[leak["Title words in descriptions"] == "removed", "CV Macro F1"].iloc[0]
warn_if(kept - removed > 0.15,
        f"macro-F1 drops {kept - removed:.2f} when title words are removed: descriptions repeat the "
        "title, so the classifier partly rediscovers the keyword rules")
print(f"[info] test set is only {meta['n_test']} roles - differences of a few points are noise")

# ---------------------------------------------------------------------------
# 5. Skill gap
# ---------------------------------------------------------------------------
print("\n-- 5. Skill gap")
from src.skill_gap import (
    eligible_categories, get_category_skill_profile, analyze_skill_gap,
    get_category_cert_profile,
)

eligible = eligible_categories(df)
check(len(eligible) > 0, f"{len(eligible)} categories have enough roles for gap analysis")
profiles = {c: get_category_skill_profile(df, c) for c in eligible}
for cat, prof in profiles.items():
    print(f"  {cat}: {len(prof)} skills in profile")
check(all(not p.empty for p in profiles.values()), "every eligible category has a non-empty profile")

target = "DevOps & SRE" if "DevOps & SRE" in profiles else eligible[0]
lookup = build_skill_lookup(df)
res_a = analyze_skill_gap(parse_user_skills("Python, AWS, Docker, Linux", lookup), profiles[target])
res_b = analyze_skill_gap(parse_user_skills("python, aws, DOCKER, linux", lookup), profiles[target])
check(res_a["matched_skills"] == res_b["matched_skills"] and res_a["coverage_pct"] == res_b["coverage_pct"],
      "gap analysis gives identical results regardless of capitalisation")
check(0 <= res_a["coverage_pct"] <= 100 and 0 <= res_a["simple_coverage_pct"] <= 100,
      "coverage percentages are within 0-100")
check(analyze_skill_gap([], profiles[target])["coverage_pct"] == 0, "empty input gives 0% coverage")
check("error" in analyze_skill_gap(["python"], pd.DataFrame(columns=["skill", "frequency", "key"])),
      "an empty profile is reported as an error, not a crash")
print(f"[info] {target}: weighted {res_a['coverage_pct']}% / simple {res_a['simple_coverage_pct']}% "
      f"| matched {res_a['matched_skills']}")
cert_prof = get_category_cert_profile(df, target, top_n=5)
print(f"[info] {target} top certs:", cert_prof["certification"].tolist() if not cert_prof.empty else "N/A")

# ---------------------------------------------------------------------------
# 6. Files
# ---------------------------------------------------------------------------
print("\n-- 6. Files")
required = ["app.py", "requirements.txt", "data/IT_Job_Roles_Skills.csv", "src/__init__.py",
            "src/preprocessing.py", "src/analysis.py", "src/nlp.py", "src/model.py", "src/skill_gap.py"]
optional = ["README.md", "Student_JobMarketAnalytics.ipynb", "Student_ProjectReport.docx",
            "tests/test_regressions.py"]
for f in required:
    check((ROOT / f).exists(), f"required file present: {f}")
for f in optional:
    warn_if(not (ROOT / f).exists(), f"optional file missing: {f}")

# ---------------------------------------------------------------------------
print()
if failures:
    print(f"=== {len(failures)} CHECK(S) FAILED ===")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("=== ALL CHECKS PASSED ===" + (f" ({len(warnings_seen)} warning(s) above)" if warnings_seen else ""))
print(f"Dataset: {stats['total_jobs']} IT roles, {df['category'].nunique()} categories, "
      f"{stats['total_unique_skills']} unique skills")
print(f"Classifier: {selected} - test accuracy {sel['Accuracy']:.1%}, macro-F1 {sel['Macro F1']:.2f} "
      f"(baseline {base['Accuracy']:.1%})")