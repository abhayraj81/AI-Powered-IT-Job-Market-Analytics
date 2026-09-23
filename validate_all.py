import sys, warnings, os
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
import nltk; nltk.download('stopwords', quiet=True)

print('=== FINAL VALIDATION (IT_Job_Roles_Skills.csv) ===')

# 1. Preprocessing
from src.preprocessing import load_and_clean, get_dataset_stats, normalize_skill
df = load_and_clean('data/IT_Job_Roles_Skills.csv')
stats = get_dataset_stats(df)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Categories:', df['category'].value_counts().to_dict())
total_jobs = stats['total_jobs']
total_skills = stats['total_unique_skills']
total_certs = stats['total_unique_certs']
print(f'[OK] preprocessing: {total_jobs} roles, {total_skills} unique skills, {total_certs} unique certs')

# 2. Analysis module
from src.analysis import (
    category_distribution, top_job_titles, top_skills_overall,
    top_skills_by_category, skill_cooccurrence, skills_per_category_heatmap,
    get_skill_counts, top_certifications_overall, cert_count_distribution
)
_, _, counts = category_distribution(df)
_, sk_df = top_skills_overall(df, n=15)
co_df = skill_cooccurrence(df, top_n=20)
_, cert_df = top_certifications_overall(df, n=10)
_, heat = skills_per_category_heatmap(df, top_n_skills=20)
import matplotlib.pyplot as plt; plt.close('all')
print('[OK] analysis: all functions passed')
print('Top 5 skills:', sk_df.head(5)['Skill'].tolist())
print('Top 5 certs:', cert_df.head(5)['Certification'].tolist())

# 3. NLP module
from src.nlp import build_tfidf_matrix, top_tfidf_terms_by_category, extract_ngrams, compute_similarity_matrix
vec, mat, _ = build_tfidf_matrix(df, text_col='job_description', max_features=3000)
tfidf_cats = top_tfidf_terms_by_category(df, vec, mat, n=5)
sim_mat = compute_similarity_matrix(mat)
bigrams = extract_ngrams(df['job_description'], n=2, top_k=20)
print(f'[OK] nlp: TF-IDF {mat.shape}, similarity {sim_mat.shape}, bigrams {len(bigrams)}')

# 4. ML module
from src.model import prepare_features, split_data, train_and_evaluate_all, train_best_model, save_model, predict_category
X, y, valid_cats = prepare_features(df)
print(f'ML-eligible categories: {valid_cats}')
X_tr, X_te, y_tr, y_te = split_data(X, y)
print(f'Train: {len(X_tr)}, Test: {len(X_te)}')
results = train_and_evaluate_all(X_tr, X_te, y_tr, y_te)
print(results.to_string(index=False))
best_pipe, y_pred, report, cats = train_best_model(X_tr, X_te, y_tr, y_te)
save_model(best_pipe)
pred = predict_category(best_pipe, 'Responsible for deploying and managing Kubernetes clusters on AWS')
print(f'Prediction demo: {pred}')
best_acc = results['Accuracy'].max()
best_model = results.loc[results['Accuracy'].idxmax(), 'Model']
print(f'[OK] model: best accuracy={best_acc:.4f} ({best_model}), pred={pred}')

# 5. Skill gap module
from src.skill_gap import get_category_skill_profile, parse_user_skills, analyze_skill_gap, get_category_cert_profile
profiles = {cat: get_category_skill_profile(df, cat) for cat in df['category'].unique()}
for cat, prof in profiles.items():
    print(f'  {cat}: {len(prof)} skills in profile')
user = parse_user_skills('Python, AWS, Docker, Linux')
result = analyze_skill_gap(user, profiles['DevOps & SRE'])
print(f'DevOps coverage with sample skills: {result["coverage_pct"]}%')
print(f'Matched: {result["matched_skills"]}')
cert_prof = get_category_cert_profile(df, 'DevOps & SRE', top_n=5)
print('DevOps top certs:', cert_prof['certification'].tolist() if not cert_prof.empty else 'N/A')

# 6. File structure
required_files = [
    'Student_JobMarketAnalytics.ipynb', 'app.py', 'requirements.txt',
    'README.md', 'Student_ProjectReport.docx',
    'data/IT_Job_Roles_Skills.csv',
    'src/preprocessing.py', 'src/analysis.py', 'src/nlp.py',
    'src/model.py', 'src/skill_gap.py', 'src/__init__.py',
]
for f in required_files:
    assert os.path.exists(f), f'MISSING: {f}'
print(f'[OK] all {len(required_files)} files present')

print()
print('=== ALL VALIDATIONS PASSED ===')
print(f'Dataset: {total_jobs} IT roles, {df["category"].nunique()} categories, {total_skills} unique skills')
print(f'ML Best Accuracy: {best_acc:.1%} ({best_model})')
