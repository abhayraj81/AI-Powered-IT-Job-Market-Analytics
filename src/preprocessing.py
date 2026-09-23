"""
preprocessing.py
================
Data loading, cleaning, and skill normalisation for the
AI-Powered IT Job Market Analytics & Skill Gap Analyzer.

Dataset: IT_Job_Roles_Skills.csv  (4 columns)
    Job Title | Job Description | Skills | Certifications

Key difference from a categorised dataset:
- There is NO pre-assigned category column.
- Categories are INFERRED from job title keywords using a deterministic
  rule-based mapping (CATEGORY_RULES).  Every decision is transparent.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Category inference rules
# Each key is a canonical category name.
# Rules are checked in order; first match wins.
# keywords are matched case-insensitively against the job title.
# ---------------------------------------------------------------------------
CATEGORY_RULES = {
    "AI / Machine Learning": [
        "artificial intelligence", "machine learning", " ai ", "ai/ml",
        "ai researcher", "ai architect", "ai software", "deep learning",
        "nlp engineer", "natural language processing", "data science",
        "computer vision", "robotics",
    ],
    "Data & Analytics": [
        "data analyst", "data architect", "data engineer", "data modeler",
        "data scientist", "data warehouse", "business intelligence", "bi analyst",
        "bi developer", "big data", "etl ", "analytics", "hadoop", "spark",
        "data quality", "data governance", "data integration", "data ops",
        "database", " dba ", "admin big data", "sql developer", "oracle sql",
        "oracle developer", "information research", "gis analyst",
        "principle engineer in data",
    ],
    "DevOps & SRE": [
        "devops", "sre", "site reliability", "devsecops",
        "build and release", "build engineer", "release engineer",
        "infrastructure engineer", "platform engineer",
        "ansible", "chef ", "puppet", "terraform", "jenkins",
        "bamboo", "gitlab", "bitbucket", "artifactory", "gerrit",
        "elk engineer", "splunk", "appd", "appdynamics",
        # Specific DevOps tooling engineers
        "docker engineer", "kubernetes engineer", "kubernetes operations",
        "kubernetes admin", "kubernetes administrator",
        "grafana engineer", "prometheus engineer", "nagios engineer",
        "new relic engineer", "nexus engineer", "nomad engineer",
        "vault engineer", "consul engineer", "envoy engineer",
        "fluentd engineer", "zabbix engineer", "datadog engineer",
        "sonarqube engineer", "fortify engineer", "coverity",
        "coverage.py", "jacoco", "junit engineer",
        "tfs engineer", "teamcity engineer", "udeploy engineer",
        "xl deploy", "packer engineer", "powershell engineer",
        "groovy engineer", "gradle engineer", "maven engineer",
        "jira engineer", "jira administrator", "confluence engineer",
        "notary engineer", "falco engineer", "istio engineer",
        "openshift engineer", "openstack engineer",
        "production support engineer", "operations engineer",
        "git engineer", "github engineer",
    ],
    "Cloud Engineering": [
        "cloud architect", "cloud engineer", "cloud developer",
        "cloud administrator", "cloud automation", "cloud migration",
        "cloud system", "cloud computing specialist", "cloud/software",
        "aws architect", "aws devops", "aws solutions",
        "azure architect", "azure devops", "azure engineer",
        "gcp", "salesforce architect", "salesforce developer",
        "salesforce engineer", "salesforce admin",
        "infrastructure architect",
    ],
    "Security": [
        "security", "cybersecurity", "pentest", "penetration",
        "soc analyst", "devsec", "appsec", "application security",
        "information security", "network security", "cloud security",
        "vulnerability", "compliance", "grc", "computer forensic",
        "identity and access", "iam specialist", "it auditor",
    ],
    "Software Development": [
        ".net developer", "android developer", "ios developer",
        "mobile app", "mobile developer", "backend", "back end",
        "front end", "frontend", "full stack", "fullstack",
        "web developer", "web engineer", "api developer",
        "java developer", "java architect", "python developer", "python architect",
        "node.js", "react", "angular", "software engineer", "software developer",
        "software architect", "software development engineer",
        "application engineer", "application developer", "blockchain",
        "embedded", "firmware", "systems engineer",
        "c# developer", "ruby on rails", "php developer",
        "javascript developer", "sharepoint developer", "mulesoft",
        "microsoft dynamics", "e-commerce developer", "wordpress developer",
        "unity developer", "mainframe developer",
        "internet of things", "iot developer",
        "programmer analyst", " programmer", "coder", " developer",
        "entry level developer", "entry level programmer",
        "junior developer", "jr developer", "lead programmer",
        "micro services", "application designer",
        "erp consultant", "sap consultant",
    ],
    "Design & UX": [
        "ux ", "ui ", "user experience", "user interface",
        "interaction designer", "visual designer", "graphic design",
        "accessibility", "information architect", "animation",
        "animator", "3d artist", "2d artist", "art director",
        "computer graphics", "game ", "game developer",
        "web designer", "front-end designer",
        "character designer", "layout artist", "storyboard artist",
        "rigging artist", "vfx artist", "motion graphics",
        "compositor", "multimedia architect",
        "graphic effects supervisor",
    ],
    "Testing & QA": [
        "qa ", "quality assurance", "test engineer", "sdet",
        "automation test", "performance test", "selenium",
        "automation specialist",
    ],
    "Networking & Infrastructure": [
        "network engineer", "network architect", "network admin",
        "cisco", "wireless", "firewall", "network specialist",
        "infrastructure admin", "systems admin", "sysadmin",
        "linux admin", "windows admin",
        "network analyst", "network infrastructure", "network operations",
        "network reliability", "internet engineer",
        "computer support specialist",
        "computer hardware engineer", "cnc programmer",
    ],
    "Management & Leadership": [
        "manager", "director", "head of", " vp ", "cto", "cio",
        "scrum master", "product manager", "project manager",
        "agile ", "program manager", "team lead", "technical lead",
        "it leader", "it director", "solutions architect",
        "enterprise architect", "it strategist",
        "chief information", "digital transformation",
        "technology officer", "technical operations officer",
        "senior it consultant", "it consultant",
        "technology assistant", "technology specialist",
        "business systems analyst",
        "computer systems analyst",
        "information technology analyst",
        "it sales", "technology sales", "tech sales",
        "seo consultant", "it apprentice", "intelligence specialist",
    ],
    "IT Support & Operations": [
        "it support", "it operations", "help desk", "service desk",
        "it administrator", "systems support", "end user",
        "deskside", "it technician", "customer service representative",
        "technical support",
    ],
}


def infer_category(title: str) -> str:
    """
    Return the inferred category for a job title using CATEGORY_RULES.
    Matching is case-insensitive; first matching category wins.
    Returns 'Other IT' if no rule matches.
    """
    t = " " + title.lower() + " "  # pad so word-boundary checks work
    for category, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in t:
                return category
    return "Other IT"


# ---------------------------------------------------------------------------
# Skill normalisation mapping
# ---------------------------------------------------------------------------
SKILL_NORMALIZATION = {
    # Cloud providers
    "amazon web services": "AWS",
    "google cloud platform": "GCP",
    "google cloud": "GCP",
    "microsoft azure": "Azure",
    # Python
    "python programming": "Python",
    "python3": "Python",
    # JavaScript
    "js": "JavaScript",
    "javascript": "JavaScript",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    # React / Angular
    "reactjs": "React",
    "react.js": "React",
    "angularjs": "Angular",
    # SQL variants
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "ms sql": "SQL Server",
    "microsoft sql server": "SQL Server",
    "t-sql": "SQL",
    "pl/sql": "SQL",
    # DevOps tools
    "ci/cd": "CI/CD",
    "ci / cd": "CI/CD",
    "continuous integration": "CI/CD",
    "continuous delivery": "CI/CD",
    "infrastructure as code": "IaC",
    "iac": "IaC",
    "k8s": "Kubernetes",
    # Communication
    "communication skills": "Communication",
    "verbal communication": "Communication",
    "written communication": "Communication",
    # Project management
    "project management skills": "Project Management",
    # Machine learning
    "ml": "Machine Learning",
    "machine learning (ml)": "Machine Learning",
    # Deep learning
    "dl": "Deep Learning",
    # Problem solving
    "problem-solving": "Problem Solving",
    "problem-solving skills": "Problem Solving",
    # Linux
    "linux/unix": "Linux",
    "unix": "Linux",
    # Git
    "github": "Git",
    "gitlab": "GitLab",
    # Excel
    "ms excel": "Excel",
    "microsoft excel": "Excel",
}


def normalize_skill(skill: str) -> str:
    """Apply normalisation map; return canonical form or title-cased original."""
    cleaned = skill.strip()
    lower = cleaned.lower()
    return SKILL_NORMALIZATION.get(lower, cleaned)


def parse_skills_csv(raw: str) -> list:
    """Parse comma-separated skills string into a normalised list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    return [normalize_skill(p) for p in parts]


def parse_certifications_csv(raw: str) -> list:
    """Parse comma-separated certifications string into a list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def clean_job_title(title: str) -> str:
    """Strip, collapse whitespace, title-case."""
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", title.strip()).title()


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load IT_Job_Roles_Skills.csv and return a clean DataFrame.

    Columns after cleaning:
        job_title       – cleaned title
        job_description – original description text
        skills          – list of normalised skills
        certifications  – list of certifications
        category        – inferred category (from CATEGORY_RULES)
        skill_count     – number of skills
        cert_count      – number of certifications
        skills_str      – comma-joined skills (for NLP/word-cloud)
        certs_str       – comma-joined certifications
    """
    path = Path(filepath)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    print(f"[Load] Raw shape: {df.shape}")

    # ── Normalise column names (strip spaces) ──────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    required = {"Job Title", "Job Description", "Skills", "Certifications"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # ── Rename to snake_case ───────────────────────────────────────────────
    df = df.rename(columns={
        "Job Title": "job_title",
        "Job Description": "job_description",
        "Skills": "skills_raw",
        "Certifications": "certifications_raw",
    })

    # ── Drop exact duplicates ──────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"[Dedup] Removed {removed} exact duplicate rows. Remaining: {len(df)}")

    # ── Handle title-level duplicates: keep first occurrence ──────────────
    # (24 titles appear twice, likely from mixed-case versions)
    before2 = len(df)
    df["_title_norm"] = df["job_title"].str.strip().str.upper()
    df = df.drop_duplicates(subset="_title_norm", keep="first")
    df = df.drop(columns=["_title_norm"])
    dup_removed = before2 - len(df)
    if dup_removed:
        print(f"[Dedup] Removed {dup_removed} title-duplicate rows. Remaining: {len(df)}")

    # ── Clean text columns ────────────────────────────────────────────────
    df["job_title"] = df["job_title"].astype(str).apply(clean_job_title)
    df["job_description"] = df["job_description"].astype(str).str.strip()
    df["skills_raw"] = df["skills_raw"].astype(str).str.strip()
    df["certifications_raw"] = df["certifications_raw"].astype(str).str.strip()

    # ── Parse skills and certifications ───────────────────────────────────
    df["skills"] = df["skills_raw"].apply(parse_skills_csv)
    df["certifications"] = df["certifications_raw"].apply(parse_certifications_csv)

    # ── Infer category ────────────────────────────────────────────────────
    df["category"] = df["job_title"].apply(infer_category)

    # ── Derived columns ───────────────────────────────────────────────────
    df["skill_count"] = df["skills"].apply(len)
    df["cert_count"] = df["certifications"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda x: ", ".join(x))
    df["certs_str"] = df["certifications"].apply(lambda x: ", ".join(x))

    print(f"[Done] Clean shape: {df.shape}")
    print(f"[Done] Categories inferred: {df['category'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return key statistics for the cleaned IT dataset."""
    all_skills = [s for skills in df["skills"] for s in skills]
    all_certs = [c for certs in df["certifications"] for c in certs]
    return {
        "total_jobs": len(df),
        "total_categories": df["category"].nunique(),
        "total_unique_titles": df["job_title"].nunique(),
        "total_unique_skills": len(set(all_skills)),
        "total_unique_certs": len(set(all_certs)),
        "avg_skills_per_job": round(df["skill_count"].mean(), 2),
        "median_skills_per_job": df["skill_count"].median(),
        "avg_certs_per_job": round(df["cert_count"].mean(), 2),
        "category_counts": df["category"].value_counts().to_dict(),
        "missing_description": (df["job_description"].str.strip() == "").sum(),
        "missing_skills": (df["skill_count"] == 0).sum(),
    }


def parse_user_skills(raw_input: str) -> list:
    """
    Parse and normalise comma-separated user skill input.
    Returns a deduplicated list of normalised skill strings.
    """
    if not raw_input or not raw_input.strip():
        return []
    parts = [p.strip() for p in raw_input.split(",") if p.strip()]
    normalised = [normalize_skill(p) for p in parts]
    seen = set()
    result = []
    for s in normalised:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result
