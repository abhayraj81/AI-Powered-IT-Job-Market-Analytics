"""
Data loading, cleaning, and skill normalisation for the
AI-Powered IT Job Market Analytics & Skill Gap Analyzer.

Dataset: IT_Job_Roles_Skills.csv  (4 columns)
    Job Title | Job Description | Skills | Certifications

There is NO category column in the dataset. Categories are INFERRED from
job-title keywords using a deterministic rule-based mapping (CATEGORY_RULES).

Matching rules (see _compile_keyword):
  * a keyword must START at a word boundary, so "ux" no longer matches inside
    "Linux" and "cto" no longer matches inside "Vector";
  * short keywords (<= 3 characters) and keywords written with padding spaces
    must also END at a word boundary ("ai" does not match "Aircraft");
  * longer keywords may continue ("network admin" matches "Network
    Administrator").
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Category inference rules
# Each key is a canonical category name. Rules are checked in order; the first
# category with a matching keyword wins. Matching is case-insensitive and is
# done against the job title only.
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


def _compile_keyword(keyword: str) -> re.Pattern:
    """Compile one rule keyword into a word-boundary-aware regex."""
    stripped = keyword.strip().lower()
    strict = keyword != keyword.strip() or len(stripped) <= 3
    pattern = r"(?<![a-z0-9])" + re.escape(stripped)
    if strict:
        pattern += r"(?![a-z0-9])"
    return re.compile(pattern)


_COMPILED_RULES = [
    (category, [_compile_keyword(k) for k in keywords])
    for category, keywords in CATEGORY_RULES.items()
]


def infer_category(title: str) -> str:
    """
    Return the inferred category for a job title using CATEGORY_RULES.
    Case-insensitive, word-boundary aware, first matching category wins.
    Returns 'Other IT' if no rule matches.
    """
    t = str(title).lower()
    for category, patterns in _COMPILED_RULES:
        if any(p.search(t) for p in patterns):
            return category
    return "Other IT"


# ---------------------------------------------------------------------------
# Skill normalisation
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


def skill_key(skill) -> str:
    """Canonical lowercase key. ALL skill comparisons in the project use this."""
    return re.sub(r"\s+", " ", str(skill).strip().lower())


def normalize_skill(skill: str) -> str:
    """Apply the normalisation map (case-insensitive); otherwise return the cleaned input."""
    cleaned = re.sub(r"\s+", " ", str(skill).strip())
    return SKILL_NORMALIZATION.get(cleaned.lower(), cleaned)


def parse_skills_csv(raw) -> list:
    """Parse a comma-separated skills string into a normalised list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    return [normalize_skill(p) for p in parts]


_JUNK_CHARS = re.compile(r"[\ufffd\x00-\x08\x0b-\x1f\x7f]")


def clean_certification(cert: str) -> str:
    """Remove replacement characters / control characters left by PDF extraction."""
    cert = _JUNK_CHARS.sub("", str(cert))
    return re.sub(r"\s+", " ", cert).strip()


def parse_certifications_csv(raw) -> list:
    """Parse a comma-separated certifications string into a list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    cleaned = (clean_certification(s) for s in raw.split(","))
    return [c for c in cleaned if c]


def clean_job_title(title) -> str:
    """Collapse whitespace; title-case only all-lowercase titles (keeps DevOps, iOS, AWS)."""
    if not isinstance(title, str):
        return ""
    t = re.sub(r"\s+", " ", title.strip())
    return t.title() if t.islower() else t


def _harmonise_case(lists: pd.Series) -> pd.Series:
    """
    Merge case variants ('python' / 'Python') into the most frequent spelling
    and drop duplicates inside each role's list (order preserved).
    """
    forms: dict = {}
    for items in lists:
        for s in items:
            forms.setdefault(skill_key(s), Counter())[s] += 1
    canonical = {
        key: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for key, counter in forms.items()
    }

    def fix(items):
        seen, out = set(), []
        for s in items:
            k = skill_key(s)
            if k not in seen:
                seen.add(k)
                out.append(canonical[k])
        return out

    return lists.apply(fix)


def build_skill_lookup(df: pd.DataFrame) -> dict:
    """{lowercase key: canonical dataset spelling} for every skill in the dataset."""
    return {skill_key(s): s for skills in df["skills"] for s in skills}


def parse_user_skills(raw_input: str, lookup: dict | None = None) -> list:
    """
    Parse and normalise user-typed skills (comma / semicolon / newline separated).

    Case-insensitive: with `lookup` (from build_skill_lookup) 'python' becomes the
    dataset's spelling 'Python'. Duplicates are removed case-insensitively.
    """
    if not raw_input or not raw_input.strip():
        return []
    result, seen = [], set()
    for part in re.split(r"[,;\n]+", raw_input):
        part = part.strip()
        if not part:
            continue
        skill = normalize_skill(part)
        key = skill_key(skill)
        if lookup and key in lookup:
            skill = lookup[key]
        if key not in seen:
            seen.add(key)
            result.append(skill)
    return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_and_clean(filepath) -> pd.DataFrame:
    """
    Load IT_Job_Roles_Skills.csv and return a clean DataFrame.

    Columns after cleaning:
        job_title       - cleaned title
        job_description - original description text
        skills          - list of normalised skills (case-harmonised, no duplicates)
        certifications  - list of certifications
        category        - inferred category (from CATEGORY_RULES)
        skill_count     - number of skills
        cert_count      - number of certifications
        skills_str      - comma-joined skills
        certs_str       - comma-joined certifications
    """
    path = Path(filepath)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    print(f"[Load] Raw shape: {df.shape}")

    df.columns = [c.strip() for c in df.columns]
    required = {"Job Title", "Job Description", "Skills", "Certifications"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df.rename(columns={
        "Job Title": "job_title",
        "Job Description": "job_description",
        "Skills": "skills_raw",
        "Certifications": "certifications_raw",
    })

    # Missing values become "" (NOT the string "nan", which would become a fake skill).
    for col in ["job_title", "job_description", "skills_raw", "certifications_raw"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["job_title"] != ""]

    before = len(df)
    df = df.drop_duplicates()
    print(f"[Dedup] Removed {before - len(df)} exact duplicate rows. Remaining: {len(df)}")

    before2 = len(df)
    df["_title_norm"] = df["job_title"].str.replace(r"\s+", " ", regex=True).str.upper()
    df = df.drop_duplicates(subset="_title_norm", keep="first").drop(columns=["_title_norm"])
    if before2 - len(df):
        print(f"[Dedup] Removed {before2 - len(df)} title-duplicate rows. Remaining: {len(df)}")

    df["job_title"] = df["job_title"].apply(clean_job_title)
    df["skills"] = _harmonise_case(df["skills_raw"].apply(parse_skills_csv))
    df["certifications"] = _harmonise_case(df["certifications_raw"].apply(parse_certifications_csv))
    df["category"] = df["job_title"].apply(infer_category)

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
        "missing_description": int((df["job_description"].str.strip() == "").sum()),
        "missing_skills": int((df["skill_count"] == 0).sum()),
    }
