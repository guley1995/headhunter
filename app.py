import os
import re
import io
import json
import html
import sqlite3
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import requests
import pdfplumber
import openai
from fpdf import FPDF

DB_PATH = "sniper_crm.db"


# =========================
# Helpers
# =========================
def now_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def today_date() -> date:
    return date.today()


def month_key(d: date) -> str:
    return d.strftime("%Y-%m")


def truncate_text(text: str, max_chars: int = 80000) -> str:
    if not text:
        return ""
    t = str(text)
    if len(t) <= max_chars:
        return t
    return t[:60000] + "\n\n--- [TRUNCATED] ---\n\n" + t[-20000:]


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, float) and pd.isna(v):
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, float) and pd.isna(v):
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s) if s else None
    except Exception:
        return None


def make_group_id(name: str) -> str:
    base = (name or "kandidat").strip().lower()
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^a-z0-9äöüß_\-]", "", base)
    return f"{base}_{now_date_str()}"


def confidence_rank(conf: str) -> int:
    c = (conf or "").lower().strip()
    return {"high": 3, "medium": 2, "low": 1}.get(c, 0)


def source_rank(src: str) -> int:
    s = (src or "").lower()
    if "google jobs" in s:
        return 5
    if "linkedin" in s:
        return 4
    if "stepstone" in s:
        return 3
    if "indeed" in s:
        return 2
    return 1


def normalize_source_label(src: str) -> str:
    s = (src or "").lower()
    if "linkedin" in s:
        return "LinkedIn"
    if "indeed" in s:
        return "Indeed"
    if "stepstone" in s:
        return "StepStone"
    if "glassdoor" in s:
        return "Glassdoor"
    if "google jobs" in s or s == "google_jobs":
        return "Google Jobs"
    return src or ""


def fix_pdf_text(t: Any) -> str:
    if t is None:
        return ""
    s = str(t)
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u2022": "-",
        "\u00a0": " ", "\u200b": "",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("cp1252", "replace").decode("cp1252")


def extract_text_from_pdfs(files: List[Any]) -> str:
    texts: List[str] = []
    for f in files:
        try:
            data = f.read()
            bio = io.BytesIO(data)
            with pdfplumber.open(bio) as pdf:
                t = "\n".join([(p.extract_text() or "") for p in pdf.pages])
            if t.strip():
                name = getattr(f, "name", "pdf")
                texts.append(f"\n\n--- DOKUMENT: {name} ---\n{t}")
        except Exception:
            continue
    return "\n".join(texts).strip()


def strip_html_to_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    txt = re.sub(r"(?is)<br\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</p>", "\n", txt)
    txt = re.sub(r"(?is)<.*?>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t\r]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


def looks_like_aggregator_title(title: str) -> bool:
    t = (title or "").lower()
    bad = [
        "jobs |", "jetzt", "finden sie", "offene stellen", "stellenangebote",
        "jobangebote", "jobs in", "stellen in", "top-jobs", "anzeigen", "suche",
        "zu besetzende", "offene jobs", "stellen finden"
    ]
    return any(b in t for b in bad)


def is_probably_job_posting(url: str, title: str, snippet: str, source: str) -> bool:
    u = (url or "").lower().strip()
    t = (title or "").lower().strip()
    sn = (snippet or "").lower().strip()
    src = (source or "").lower().strip()

    if "google jobs" in src:
        return True

    if not u or len(u) < 10:
        return False

    if looks_like_aggregator_title(t):
        if not any(x in u for x in ["jobs/view", "viewjob", "stellenangebote", "job-listing", "jk="]):
            return False

    if "linkedin" in src:
        return ("linkedin.com/jobs" in u) and (("jobs/view" in u) or ("currentjobid" in u) or ("jobs/search" in u))
    if "indeed" in src:
        return ("indeed" in u) and (("viewjob" in u) or ("jk=" in u) or ("clk" in u) or ("/jobs?" in u))
    if "stepstone" in src:
        return ("stepstone" in u) and (("stellenangebote" in u) or ("job" in u))
    if "glassdoor" in src:
        return ("glassdoor" in u) and (("job-listing" in u) or ("joblisting" in u))

    return True


def dedupe_jobs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        link = (it.get("link") or "").strip().lower()
        key = link if link else ((it.get("source", "") + "|" + it.get("title", "") + "|" + it.get("snippet", ""))[:260].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def score_sort_key(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
    strict_mp = safe_int(item.get("_match_percent", -1), -1)
    raw_mp = safe_int(item.get("_raw_match_percent", -1), -1)
    cr = safe_int(item.get("_conf_rank", 0), 0)
    sr = safe_int(item.get("_src_rank", 0), 0)
    return (strict_mp, raw_mp, cr, sr)


def get_workspace_id() -> str:
    return st.session_state.get("workspace_id", "default")


# =========================
# API Keys
# =========================
def get_api_keys() -> Tuple[str, str]:
    oa = st.session_state.get("OPENAI_API_KEY", "")
    sa = st.session_state.get("SERPAPI_API_KEY", "")
    if oa and sa:
        return oa, sa

    oa = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    sa = st.secrets.get("SERPAPI_API_KEY", "") if hasattr(st, "secrets") else ""
    if oa and sa:
        return oa, sa

    return os.getenv("OPENAI_API_KEY", ""), os.getenv("SERPAPI_API_KEY", "")


def get_client(oa_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=oa_key)


# =========================
# DB
# =========================
def db_execute(sql: str, params: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(sql, params)
    conn.commit()
    conn.close()


def db_fetch_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS leads (id INTEGER PRIMARY KEY AUTOINCREMENT)""")

    expected_columns = {
        "workspace_id": "TEXT",
        "candidate_group_id": "TEXT",
        "workflow_order": "INTEGER",

        "status": "TEXT",
        "datum": "TEXT",
        "last_action_date": "TEXT",

        "kandidat_name": "TEXT",
        "gehalt": "TEXT",
        "location": "TEXT",
        "skills": "TEXT",

        "expose_text": "TEXT",
        "berater_note": "TEXT",

        "candidate_docs_text": "TEXT",
        "dossier_json": "TEXT",

        "firma": "TEXT",
        "position": "TEXT",
        "link": "TEXT",
        "source": "TEXT",
        "job_description": "TEXT",

        "job_req_json": "TEXT",
        "match_json": "TEXT",

        "match_percent": "INTEGER",
        "raw_match_percent": "INTEGER",

        "match_confidence": "TEXT",
        "match_strengths": "TEXT",
        "match_gaps": "TEXT",
        "hot_reason": "TEXT",

        "provision": "REAL",
        "mail_sent_count": "INTEGER",
        "mail_last_sent_date": "TEXT",
    }

    c.execute("PRAGMA table_info(leads)")
    existing_cols = [info[1] for info in c.fetchall()]
    for col, col_type in expected_columns.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE leads ADD COLUMN {col} {col_type}")

    c.execute("""
        CREATE TABLE IF NOT EXISTS placements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            lead_id INTEGER,
            candidate_group_id TEXT,
            kandidat_name TEXT,
            firma TEXT,
            position TEXT,
            fee_total REAL,
            currency TEXT,
            share_percent REAL,
            share_min REAL,
            share_amount REAL,
            placed_at TEXT,
            invoice_status TEXT,
            invoice_note TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS lead_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            lead_id INTEGER,
            action TEXT,
            action_date TEXT,
            meta_json TEXT
        )
    """)

    conn.commit()
    conn.close()

    db_execute("UPDATE leads SET mail_sent_count = COALESCE(mail_sent_count,0) WHERE mail_sent_count IS NULL")
    db_execute("UPDATE leads SET workspace_id = COALESCE(workspace_id,'default') WHERE workspace_id IS NULL OR workspace_id = ''")
    db_execute("UPDATE leads SET raw_match_percent = COALESCE(raw_match_percent, -1) WHERE raw_match_percent IS NULL")


def load_df(workspace_id: str) -> pd.DataFrame:
    return db_fetch_df("SELECT * FROM leads WHERE workspace_id=? ORDER BY id DESC", (workspace_id,))


def log_action(workspace_id: str, lead_id: int, action: str, meta: Optional[dict] = None):
    db_execute(
        "INSERT INTO lead_actions (workspace_id, lead_id, action, action_date, meta_json) VALUES (?, ?, ?, ?, ?)",
        (workspace_id, lead_id, action, now_date_str(), safe_json_dumps(meta or {}))
    )


def update_status(workspace_id: str, lead_id: int, new_status: str):
    db_execute(
        "UPDATE leads SET status=?, last_action_date=? WHERE id=? AND workspace_id=?",
        (new_status, now_date_str(), lead_id, workspace_id)
    )
    log_action(workspace_id, lead_id, "status_change", {"status": new_status})


def delete_lead(workspace_id: str, lead_id: int):
    db_execute("DELETE FROM leads WHERE id=? AND workspace_id=?", (lead_id, workspace_id))
    db_execute("DELETE FROM lead_actions WHERE lead_id=? AND workspace_id=?", (lead_id, workspace_id))


def increment_mail_sent(workspace_id: str, lead_id: int):
    db_execute(
        "UPDATE leads SET mail_sent_count = COALESCE(mail_sent_count,0)+1, mail_last_sent_date=? WHERE id=? AND workspace_id=?",
        (now_date_str(), lead_id, workspace_id),
    )
    log_action(workspace_id, lead_id, "mail_sent_counter_inc", {})


def save_candidate_row(
    workspace_id: str,
    candidate_group_id: str,
    name: str,
    expose: str,
    note: str,
    gehalt: str,
    location: str,
    skills: str,
    docs_text: str,
    dossier_json: str,
):
    datum = now_date_str()
    db_execute(
        """INSERT INTO leads
           (workspace_id, candidate_group_id, status, datum, last_action_date,
            kandidat_name, expose_text, berater_note, gehalt, location, skills,
            candidate_docs_text, dossier_json,
            firma, position, source,
            mail_sent_count, match_percent, raw_match_percent)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, -1, -1)""",
        (
            workspace_id, candidate_group_id, "Screening", datum, datum,
            name, expose, note, gehalt, location, skills,
            docs_text, dossier_json,
            "Noch offen", "In Analyse", "Screening",
        )
    )


def save_job_lead_from_queue(workspace_id: str, search: dict, job: dict, workflow_order: int):
    src = normalize_source_label(job.get("source", ""))
    title = job.get("title", "")
    company = job.get("company", "") or "Unbekannt"
    link = job.get("link", "")
    snippet = job.get("snippet", "") or ""

    match = job.get("_match") or {}
    strict_mp = safe_int(match.get("strict_match_percent", match.get("match_percent", -1)), -1)
    raw_mp = safe_int(match.get("raw_match_percent", -1), -1)
    conf = (match.get("confidence") or "").lower().strip()

    hot_threshold = safe_int(st.session_state.get("hot_threshold", 90), 90)
    min_conf = st.session_state.get("min_conf", "medium")
    conf_ok = (conf == "high") if min_conf == "high" else (conf in ["medium", "high"])
    hot_flag = bool(match.get("over_90_criteria_met", False) and strict_mp >= hot_threshold and conf_ok)

    status = "Hot" if hot_flag else "Offen"
    hot_reason = match.get("why_over_90", "") if hot_flag else ""

    strengths = match.get("strengths", [])
    gaps = match.get("gaps", [])

    datum = now_date_str()
    db_execute(
        """INSERT INTO leads
           (workspace_id, candidate_group_id, workflow_order,
            status, datum, last_action_date,
            kandidat_name, gehalt, location, skills,
            expose_text, berater_note,
            candidate_docs_text, dossier_json,
            firma, position, link, source, job_description,
            job_req_json, match_json,
            match_percent, raw_match_percent, match_confidence, match_strengths, match_gaps, hot_reason,
            mail_sent_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            workspace_id,
            search.get("candidate_group_id", ""),
            workflow_order,
            status, datum, datum,
            search.get("name", ""),
            search.get("gehalt", ""),
            search.get("location", ""),
            search.get("skills", ""),
            search.get("expose", ""),
            search.get("note", ""),
            search.get("docs_text", ""),
            search.get("dossier_json", ""),
            company, title, link, src, snippet,
            safe_json_dumps(job.get("_job_req", {})),
            safe_json_dumps(match),
            strict_mp,
            raw_mp,
            match.get("confidence", ""),
            safe_json_dumps(strengths),
            safe_json_dumps(gaps),
            hot_reason,
        )
    )


# =========================
# Export
# =========================
def export_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    for engine in ["openpyxl", "xlsxwriter"]:
        try:
            with pd.ExcelWriter(output, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name="Sniper")
            return output.getvalue()
        except ModuleNotFoundError:
            output = io.BytesIO()
            continue
        except Exception:
            output = io.BytesIO()
            continue
    return df.to_csv(index=False).encode("utf-8")


# =========================
# PDF
# =========================
class ExposePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(40, 70, 120)
        self.cell(0, 10, fix_pdf_text("KANDIDATEN-EXPOSÉ"), 0, 1, "C")
        self.set_font("Arial", "I", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, fix_pdf_text("HG Mediadesign - office@hg-mediadesign.de"), 0, 1, "C")
        self.ln(4)
        self.line(10, 28, 200, 28)
        self.ln(10)


def create_candidate_pdf(row: dict) -> bytes:
    pdf = ExposePDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, fix_pdf_text(f"Kandidat: {row.get('kandidat_name','')}"), 0, 1)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, fix_pdf_text(f"Ort: {row.get('location','')} | Gehalt: {row.get('gehalt','')}"), 0, 1)

    strict_mp = safe_int(row.get("match_percent"), -1)
    raw_mp = safe_int(row.get("raw_match_percent"), -1)
    if strict_mp >= 0 or raw_mp >= 0:
        pdf.cell(0, 7, fix_pdf_text(f"Match: strict {strict_mp}% | raw {raw_mp}% ({row.get('match_confidence','')})"), 0, 1)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, fix_pdf_text("Profil (anonymisiert):"), 0, 1)

    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, fix_pdf_text(row.get("expose_text", "") or ""))

    note = row.get("berater_note", "") or ""
    if note.strip():
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, fix_pdf_text("Berater-Notiz:"), 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, fix_pdf_text(note))

    return pdf.output(dest="S").encode("cp1252", "replace")


# =========================
# SerpApi + Fetch
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def serpapi_get_cached(params_tuple: Tuple[Tuple[str, str], ...]) -> Dict[str, Any]:
    params = dict(params_tuple)
    return requests.get("https://serpapi.com/search.json", params=params, timeout=30).json()


def serpapi_get(params: Dict[str, Any]) -> Dict[str, Any]:
    params_tuple = tuple(sorted((str(k), str(v)) for k, v in params.items()))
    return serpapi_get_cached(params_tuple)


def fetch_google_jobs(sa_key: str, q: str, location: str, hl="de", gl="de") -> List[Dict[str, Any]]:
    params = {"engine": "google_jobs", "q": q, "location": location, "api_key": sa_key, "hl": hl, "gl": gl}
    res = serpapi_get(params)
    jobs = res.get("jobs_results", []) or []
    out = []
    for j in jobs:
        out.append({
            "source": "Google Jobs",
            "title": j.get("title", "") or "",
            "company": j.get("company_name", "") or "Unbekannt",
            "location": j.get("location", "") or location,
            "link": j.get("link", "") or "",
            "snippet": (j.get("description") or j.get("snippet") or "")[:1500],
            "raw": j,
        })
    return out


def fetch_site_jobs(
    sa_key: str,
    q: str,
    location: str,
    site_query: str,
    label: str,
    radius_hint: str = "",
    hl="de",
    gl="de",
    num=10
) -> List[Dict[str, Any]]:
    query = f'{site_query} "{q}" "{location}"'
    if radius_hint:
        query += f" {radius_hint}"
    params = {"engine": "google", "q": query, "api_key": sa_key, "hl": hl, "gl": gl}
    res = serpapi_get(params)
    organic = res.get("organic_results", []) or []
    out: List[Dict[str, Any]] = []
    for r in organic[:num]:
        out.append({
            "source": normalize_source_label(label),
            "title": r.get("title", "") or "",
            "company": "Unbekannt",
            "location": location,
            "link": r.get("link", "") or "",
            "snippet": (r.get("snippet", "") or "")[:1500],
            "raw": r,
        })
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_job_page_text(url: str) -> str:
    if not url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (SniperRecruitingBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code >= 400:
            return ""
        return strip_html_to_text(r.text)
    except Exception:
        return ""


# =========================
# AI
# =========================
def analyze_candidate_profile(oa_key: str, docs_text: str) -> dict:
    client = get_client(oa_key)
    prompt = """
Analysiere die Bewerber-Unterlagen (DE). Nutze ALLE Dokumente (CV, Zeugnisse, Zertifikate).
Gib JSON zurück:

{
 "name":"",
 "job_titel":"",
 "gehalt":"",
 "location":"",
 "skills":"",
 "expose":""
}

Regeln:
- "skills" als kommagetrennte Tags (max 20)
- "expose" 6-10 Sätze, anonymisiert (keine Firmennamen/Emails/Telefon)
- Keine Halluzination: wenn unklar, leer lassen
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + "\n\n" + truncate_text(docs_text, 70000)}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def build_candidate_dossier(oa_key: str, docs_text: str) -> dict:
    client = get_client(oa_key)
    prompt = """
Erstelle ein Kandidaten-Dossier (DE) als JSON aus ALLEN Dokumenten.
Nutze CV + Zeugnisse + Zertifikate. Keine PII.

JSON:
{
 "name_label":"",
 "target_titles":[],
 "seniority":"",
 "locations":[],
 "salary_expectation":"",
 "core_skills":[],
 "tools_tech":[],
 "certificates":[],
 "education":[],
 "experience_summary":"",
 "projects":[],
 "industries":[],
 "languages":[],
 "proof_points":[]
}
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + "\n\n" + truncate_text(docs_text, 70000)}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def extract_job_requirements_cached(oa_key: str, job_text: str) -> dict:
    client = get_client(oa_key)
    prompt = """
Extrahiere aus dem Jobtext strukturierte Anforderungen als JSON.

JSON:
{
 "title":"",
 "seniority":"",
 "location":"",
 "must_haves":[{"req":"", "weight":1-5}],
 "nice_to_haves":[{"req":"", "weight":1-5}],
 "constraints":["z.B. onsite, security clearance, language, travel"],
 "keywords":[]
}
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + "\n\n" + truncate_text(job_text, 22000)}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def get_match_v2(oa_key: str, candidate_dossier: dict, job_req: dict, job_text: str) -> dict:
    client = get_client(oa_key)
    prompt = f"""
Du bist Lead-Recruiter. Gib ZWEI Scores:

1) raw_match_percent (0-100):
   - semantische Passung basierend auf Skills/Titel/Branche
   - darf 90+ sein, auch wenn nicht jede Evidenz perfekt ist

2) strict_match_percent (0-100):
   - 90+ nur wenn:
     a) Must-Haves weight>=4 sind erfüllt
     b) keine Constraint-Blocker
     c) Evidence für Top-Must-Haves erkennbar

Gib JSON:
{{
 "raw_match_percent":0-100,
 "strict_match_percent":0-100,
 "confidence":"low|medium|high",
 "must_have_coverage":[{{"req":"", "covered":true/false, "evidence":"kurzer Beleg aus Dossier"}}],
 "constraint_blockers":[],
 "strengths":["..."],
 "gaps":["..."],
 "risk_flags":["..."],
 "explanation":"max 4 Sätze",
 "over_90_criteria_met": true/false,
 "why_over_90": "nur wenn over_90_criteria_met true"
}}

KANDIDAT:
{json.dumps(candidate_dossier, ensure_ascii=False)}

JOB-REQUIREMENTS:
{json.dumps(job_req, ensure_ascii=False)}

JOBTEXT:
{truncate_text(job_text, 16000)}
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)

    raw_mp = safe_int(data.get("raw_match_percent", 0), 0)
    strict_mp = safe_int(data.get("strict_match_percent", data.get("match_percent", 0)), 0)

    if strict_mp >= 90 and not bool(data.get("over_90_criteria_met", False)):
        strict_mp = 89
        data["why_over_90"] = ""

    data["raw_match_percent"] = raw_mp
    data["strict_match_percent"] = strict_mp
    data["match_percent"] = strict_mp
    return data


# =========================
# Placements
# =========================
def upsert_placement(
    workspace_id: str,
    lead_row: dict,
    fee_total: float,
    currency: str,
    share_percent: float,
    share_min: float,
    invoice_status: str,
    invoice_note: str,
):
    share_amount = max(fee_total * (share_percent / 100.0), share_min)

    lead_id = safe_int(lead_row.get("id"), 0)
    group_id = lead_row.get("candidate_group_id", "")
    kandidat_name = lead_row.get("kandidat_name", "")
    firma = lead_row.get("firma", "")
    position = lead_row.get("position", "")

    existing = db_fetch_df("SELECT id FROM placements WHERE workspace_id=? AND lead_id=?", (workspace_id, lead_id))
    if not existing.empty:
        pid = int(existing.iloc[0]["id"])
        db_execute(
            """UPDATE placements
               SET fee_total=?, currency=?, share_percent=?, share_min=?, share_amount=?,
                   invoice_status=?, invoice_note=?, placed_at=?
               WHERE id=? AND workspace_id=?""",
            (fee_total, currency, share_percent, share_min, share_amount, invoice_status, invoice_note, now_date_str(), pid, workspace_id),
        )
    else:
        db_execute(
            """INSERT INTO placements
               (workspace_id, lead_id, candidate_group_id, kandidat_name, firma, position,
                fee_total, currency, share_percent, share_min, share_amount,
                placed_at, invoice_status, invoice_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                workspace_id, lead_id, group_id, kandidat_name, firma, position,
                fee_total, currency, share_percent, share_min, share_amount,
                now_date_str(), invoice_status, invoice_note
            )
        )

    db_execute("UPDATE leads SET provision=? WHERE id=? AND workspace_id=?", (fee_total, lead_id, workspace_id))
    log_action(workspace_id, lead_id, "placement_upsert", {
        "fee_total": fee_total, "currency": currency, "share_percent": share_percent,
        "share_min": share_min, "share_amount": share_amount, "invoice_status": invoice_status
    })


def load_placements_df(workspace_id: str) -> pd.DataFrame:
    return db_fetch_df("SELECT * FROM placements WHERE workspace_id=? ORDER BY placed_at DESC, id DESC", (workspace_id,))


def load_actions_df(workspace_id: str, lead_id: int) -> pd.DataFrame:
    return db_fetch_df(
        "SELECT action_date, action, meta_json FROM lead_actions WHERE workspace_id=? AND lead_id=? ORDER BY id DESC",
        (workspace_id, lead_id)
    )


# =========================
# UI: Settings
# =========================
def render_settings():
    st.header("⚙️ Einstellungen")

    st.subheader("SaaS / Workspace")
    ws = st.text_input("Workspace ID (default: default)", value=st.session_state.get("workspace_id", "default"))
    if st.button("Workspace setzen", use_container_width=True):
        st.session_state["workspace_id"] = ws.strip() or "default"
        st.success(f"Workspace gesetzt: {st.session_state['workspace_id']}")

    st.divider()

    oa, sa = get_api_keys()
    c1, c2 = st.columns(2)
    c1.metric("OpenAI Key verfügbar", "Ja" if bool(oa) else "Nein")
    c2.metric("SerpApi Key verfügbar", "Ja" if bool(sa) else "Nein")

    st.caption("Keys: st.secrets / ENV. Optional Session-Override.")

    with st.form("settings_form"):
        override = st.toggle("Session-Override aktivieren", value=False)
        oa_in = st.text_input("OpenAI API Key (nur Session)", type="password",
                              value="" if not override else st.session_state.get("OPENAI_API_KEY", ""))
        sa_in = st.text_input("SerpApi Key (nur Session)", type="password",
                              value="" if not override else st.session_state.get("SERPAPI_API_KEY", ""))
        ok = st.form_submit_button("Speichern")

    if ok:
        if override:
            if not oa_in or not sa_in:
                st.error("Bitte beide Keys eingeben oder Override deaktivieren.")
                return
            st.session_state["OPENAI_API_KEY"] = oa_in
            st.session_state["SERPAPI_API_KEY"] = sa_in
            st.success("Session-Keys gesetzt.")
        else:
            st.session_state.pop("OPENAI_API_KEY", None)
            st.session_state.pop("SERPAPI_API_KEY", None)
            st.success("Session-Override entfernt.")

    st.divider()
    if st.button("Cache leeren (Scan/Pages)", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache geleert.")


# =========================
# UI: Dashboard
# =========================
def render_dashboard(workspace_id: str):
    st.header("📌 Dashboard")

    df = load_df(workspace_id)
    plc = load_placements_df(workspace_id)

    if df.empty:
        st.info("Noch keine Daten. Starte mit Screening.")
        return

    hot = df[df["status"] == "Hot"]
    placement = df[df["status"] == "Placement"]

    total_fee = plc["fee_total"].fillna(0).sum() if not plc.empty else 0.0
    total_share = plc["share_amount"].fillna(0).sum() if not plc.empty else 0.0

    a, b, c, d = st.columns(4)
    a.metric("🔥 Hot Leads", int(len(hot)))
    b.metric("✅ Placements", int(len(placement)))
    c.metric("💶 Fees (Placement)", f"{float(total_fee):,.2f} €")
    d.metric("💰 Dein Anteil", f"{float(total_share):,.2f} €")

    st.divider()
    st.subheader("Pipeline Funnel")
    statuses = ["Screening", "Hot", "Offen", "Kontaktiert", "Interview", "Placement", "Abgelehnt"]
    counts = {s: int((df["status"] == s).sum()) for s in statuses}
    funnel_df = pd.DataFrame([{"Status": s, "Count": counts[s]} for s in statuses])
    st.dataframe(funnel_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Letzte Placements")
    if plc.empty:
        st.info("Noch keine Placements erfasst.")
    else:
        show = plc[["placed_at", "kandidat_name", "firma", "position", "fee_total", "share_amount", "invoice_status"]].head(10)
        st.dataframe(show, use_container_width=True, hide_index=True)


# =========================
# UI: Screening
# =========================
def render_screening(workspace_id: str, oa_key: str):
    st.header("👤 Screening & CV Analyse")

    files = st.file_uploader(
        "Bewerber-Dokumente hochladen (CV, Zeugnisse, Zertifikate) – mehrere PDFs möglich",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("🔄 Screening Reset", use_container_width=True):
        for k in ["docs_text", "screening_data", "dossier_json", "active_search", "candidate_group_id"]:
            st.session_state.pop(k, None)
        st.success("Reset erledigt.")
        st.rerun()

    if not files:
        st.info("Bitte PDFs hochladen.")
        return

    if "docs_text" not in st.session_state:
        with st.spinner("Extrahiere Text aus PDFs..."):
            st.session_state.docs_text = extract_text_from_pdfs(files)

    docs_text = st.session_state.docs_text
    if not docs_text:
        st.error("Konnte keinen Text extrahieren. (Scan-PDF?)")
        return

    st.subheader("🤖 Automatische Analyse")
    colA, colB = st.columns([1, 1])
    if colA.button("Profil automatisch erzeugen", use_container_width=True):
        with st.spinner("Analysiere Unterlagen..."):
            st.session_state.screening_data = analyze_candidate_profile(oa_key, docs_text)
        st.success("Analyse erstellt.")

    if colB.button("Analyse zurücksetzen", use_container_width=True):
        st.session_state.pop("screening_data", None)
        st.success("Analyse zurückgesetzt.")
        st.rerun()

    d = st.session_state.get("screening_data", {}) or {}

    st.divider()
    st.subheader("Kandidat (editierbar)")
    c1, c2 = st.columns(2)
    name = c1.text_input("Kandidat Label/Name (intern)", value=d.get("name", ""))
    target_job = c1.text_input("Ziel-Jobtitel (für Scan)", value=d.get("job_titel", ""))
    location = c2.text_input("Ort/Region", value=d.get("location", ""))
    salary = c2.text_input("Gehaltswunsch", value=d.get("gehalt", ""))
    skills = st.text_input("Skills/Tags", value=d.get("skills", ""))
    note = st.text_area("Berater-Notiz", value="", height=120)
    expose = st.text_area("Exposé (Deutsch, anonymisiert)", value=d.get("expose", ""), height=220)

    group_id = st.text_input("Workflow-ID (auto, wenn leer)", value=st.session_state.get("candidate_group_id", ""))
    if not group_id.strip():
        group_id = make_group_id(name)
    st.session_state.candidate_group_id = group_id

    st.divider()
    st.subheader("📦 Dossier (aus allen Dokumenten)")
    if st.button("Dossier erstellen/aktualisieren", use_container_width=True):
        with st.spinner("Erstelle Dossier..."):
            dossier = build_candidate_dossier(oa_key, docs_text)
            st.session_state.dossier_json = safe_json_dumps(dossier)
        st.success("Dossier erstellt.")

    if st.session_state.get("dossier_json"):
        with st.expander("Dossier ansehen (JSON)"):
            st.json(safe_json_loads(st.session_state.dossier_json) or {})

    st.divider()
    if st.button("✅ Kandidat speichern & Scan vorbereiten", use_container_width=True):
        dossier_json = st.session_state.get("dossier_json", "")
        if not dossier_json:
            with st.spinner("Kein Dossier vorhanden → erstelle Dossier..."):
                dossier = build_candidate_dossier(oa_key, docs_text)
                dossier_json = safe_json_dumps(dossier)
                st.session_state.dossier_json = dossier_json

        save_candidate_row(
            workspace_id=workspace_id,
            candidate_group_id=group_id,
            name=name,
            expose=expose,
            note=note,
            gehalt=salary,
            location=location,
            skills=skills,
            docs_text=docs_text,
            dossier_json=dossier_json,
        )

        st.session_state.active_search = {
            "q": target_job,
            "l": location,
            "name": name,
            "expose": expose,
            "note": note,
            "gehalt": salary,
            "location": location,
            "skills": skills,
            "docs_text": docs_text,
            "dossier_json": dossier_json,
            "candidate_group_id": group_id,
        }

        st.success("Gespeichert. Weiter zu Markt-Scan.")


# =========================
# UI: Candidate selector
# =========================
def load_latest_candidate_into_session(workspace_id: str) -> bool:
    df = load_df(workspace_id)
    if df.empty:
        return False
    screen = df[df["source"].fillna("") == "Screening"].copy()
    if screen.empty:
        return False

    screen["label"] = screen.apply(
        lambda r: f"{r.get('kandidat_name','')} | {r.get('location','')} | {r.get('datum','')} | {r.get('candidate_group_id','')}",
        axis=1
    )
    choice = st.selectbox("Kandidat laden (aus DB)", screen["label"].tolist())
    row = screen[screen["label"] == choice].iloc[0].to_dict()

    st.session_state.active_search = {
        "q": "",
        "l": row.get("location", ""),
        "name": row.get("kandidat_name", ""),
        "expose": row.get("expose_text", ""),
        "note": row.get("berater_note", ""),
        "gehalt": row.get("gehalt", ""),
        "location": row.get("location", ""),
        "skills": row.get("skills", ""),
        "docs_text": row.get("candidate_docs_text", ""),
        "dossier_json": row.get("dossier_json", ""),
        "candidate_group_id": row.get("candidate_group_id", ""),
    }
    st.success("Kandidat geladen.")
    return True


# =========================
# UI: Markt-Scan
# =========================
def render_market_scan(workspace_id: str, oa_key: str, sa_key: str):
    st.header("🔍 Markt-Scan")

    if "active_search" not in st.session_state:
        st.info("Kein aktiver Kandidat in Session. Lade einen Kandidaten aus der DB:")
        ok = load_latest_candidate_into_session(workspace_id)
        if not ok:
            st.warning("Noch keine Kandidaten gespeichert. Erst Screening machen.")
            return

    search = st.session_state.active_search
    dossier = safe_json_loads(search.get("dossier_json", "") or "")

    st.caption(f"Kandidat: {search.get('name','')} | Workflow: {search.get('candidate_group_id','')}")

    c1, c2 = st.columns(2)
    q = c1.text_input("Job-Titel (Komma möglich)", value=search.get("q", ""))
    l = c2.text_input("Standort (Zentrum)", value=search.get("l", ""))

    r1, r2 = st.columns([1, 2])
    radius_km = r1.slider("Umkreis (km)", 0, 150, 50, 5)
    radius_mode = r2.selectbox("Radius-Strategie", ["Query erweitern (empfohlen)", "Aus (0 km)"],
                               index=0 if radius_km > 0 else 1)

    radius_hint_web = ""
    radius_hint_jobs = ""
    if radius_km > 0 and "Query" in radius_mode:
        radius_hint_web = f' ("Umkreis {radius_km} km" OR "within {radius_km} km" OR "{radius_km} km")'
        radius_hint_jobs = f" Umkreis {radius_km} km"

    if radius_km > 0 and "Query" in radius_mode:
        st.info(f"📍 Radius aktiv: {radius_km} km um {l}")

    st.caption("Quellen")
    s1, s2, s3, s4 = st.columns(4)
    use_google_jobs = s1.toggle("Google Jobs", value=True)
    use_linkedin = s2.toggle("LinkedIn", value=True)
    use_indeed = s3.toggle("Indeed", value=True)
    use_stepstone = s4.toggle("StepStone", value=True)

    st.divider()
    a, b, c = st.columns(3)
    max_results = a.slider("Max Ergebnisse (gesamt)", 10, 120, 40, 10)
    per_board = b.slider("Max pro Quelle", 5, 30, 10, 5)
    only_real = c.toggle("✅ Nur echte Einzelanzeigen (relaxed)", value=True)

    st.divider()
    st.subheader("Matching v2.1")
    auto_match = st.toggle("Auto-Match berechnen", value=True)
    fetch_pages = st.toggle("🔎 Job-Seite laden (stark empfohlen)", value=True)

    hot_threshold = st.slider("Hot Threshold (strict %)", 80, 99, 90, 1)
    min_conf = st.selectbox("Min Confidence für Hot", ["medium", "high"], index=0)

    st.session_state.hot_threshold = hot_threshold
    st.session_state.min_conf = min_conf

    if auto_match and not dossier:
        st.warning("Dossier fehlt. Im Screening zuerst 'Dossier erstellen' und speichern.")

    if st.button("🚀 Scan starten", use_container_width=True):
        queries = [x.strip() for x in (q or "").split(",") if x.strip()]
        if not queries:
            st.warning("Bitte mindestens einen Jobtitel eingeben.")
            return

        all_jobs: List[Dict[str, Any]] = []
        with st.spinner("Suche Jobs..."):
            for qi in queries:
                if use_google_jobs:
                    all_jobs += fetch_google_jobs(sa_key, qi + radius_hint_jobs, l)

                if use_linkedin:
                    all_jobs += fetch_site_jobs(
                        sa_key, qi, l,
                        'site:linkedin.com/jobs',
                        "LinkedIn",
                        radius_hint=radius_hint_web,
                        num=per_board
                    )
                if use_indeed:
                    all_jobs += fetch_site_jobs(
                        sa_key, qi, l,
                        '(site:indeed.com OR site:indeed.de)',
                        "Indeed",
                        radius_hint=radius_hint_web,
                        num=per_board
                    )
                if use_stepstone:
                    all_jobs += fetch_site_jobs(
                        sa_key, qi, l,
                        'site:stepstone.de',
                        "StepStone",
                        radius_hint=radius_hint_web,
                        num=per_board
                    )

        all_jobs = dedupe_jobs(all_jobs)
        all_jobs_raw = list(all_jobs)

        if only_real:
            kept = []
            for j in all_jobs:
                if normalize_source_label(j.get("source", "")) == "Google Jobs":
                    kept.append(j)
                    continue
                if is_probably_job_posting(j.get("link", ""), j.get("title", ""), j.get("snippet", ""), j.get("source", "")):
                    kept.append(j)
            all_jobs = kept

        if not all_jobs:
            st.warning("Filter war zu streng → zeige ungefilterte Treffer (deaktiviere Filter für mehr).")
            all_jobs = all_jobs_raw

        if not all_jobs:
            st.warning("Keine Ergebnisse. Jobtitel/Standort ändern.")
            return

        enriched: List[Dict[str, Any]] = []
        with st.spinner("Matching..."):
            for j in all_jobs:
                src = normalize_source_label(j.get("source", ""))
                j["source"] = src
                j["_src_rank"] = source_rank(src)

                base_text = f"{j.get('title','')}\n{j.get('snippet','')}".strip()
                need_page = (len(base_text) < 600)

                if (fetch_pages or need_page) and j.get("link"):
                    page_txt = fetch_job_page_text(j["link"])
                    if page_txt:
                        base_text = base_text + "\n\n[PAGE]\n" + page_txt[:30000]

                j["_job_text"] = base_text
                j["_job_req"] = {}
                j["_match"] = {}
                j["_match_percent"] = -1
                j["_raw_match_percent"] = -1
                j["_conf_rank"] = 0
                j["_hot"] = False

                if auto_match and dossier:
                    try:
                        job_req = extract_job_requirements_cached(oa_key, base_text)
                        match = get_match_v2(oa_key, dossier, job_req, base_text)

                        strict_mp = safe_int(match.get("strict_match_percent", match.get("match_percent", -1)), -1)
                        raw_mp = safe_int(match.get("raw_match_percent", -1), -1)
                        conf = (match.get("confidence") or "").lower().strip()

                        conf_ok = (conf == "high") if min_conf == "high" else (conf in ["medium", "high"])
                        hot_flag = bool(match.get("over_90_criteria_met", False) and strict_mp >= hot_threshold and conf_ok)

                        j["_job_req"] = job_req
                        j["_match"] = match
                        j["_match_percent"] = strict_mp
                        j["_raw_match_percent"] = raw_mp
                        j["_conf_rank"] = confidence_rank(conf)
                        j["_hot"] = hot_flag
                    except Exception:
                        pass

                enriched.append(j)

        enriched.sort(key=score_sort_key, reverse=True)
        enriched = enriched[:max_results]

        st.session_state.scan_results = enriched
        st.success(f"{len(enriched)} Ergebnisse in Review-Queue gelegt.")
        st.info("Weiter zu: Review-Queue")


# =========================
# UI: Review Queue
# =========================
def render_review_queue(workspace_id: str):
    st.header("🧾 Review-Queue")

    results = st.session_state.get("scan_results", [])
    if not results:
        st.info("Keine Scan-Ergebnisse. Starte einen Scan.")
        return

    if "active_search" not in st.session_state:
        st.warning("Kein aktiver Kandidat in Session. Im Markt-Scan Kandidat laden.")
        return

    search = st.session_state.active_search

    c1, c2, c3 = st.columns(3)
    sort_mode = c1.selectbox("Sortierung", ["Bester Match (strict)", "Bester Match (raw)", "Quelle", "Titel"], index=0)
    show_only_hot = c2.toggle("🔥 Nur Hot (strict)", value=False)
    bulk = c3.toggle("Bulk-Auswahl", value=True)

    items = list(results)
    if show_only_hot:
        items = [x for x in items if x.get("_hot")]

    if sort_mode == "Bester Match (strict)":
        items.sort(key=score_sort_key, reverse=True)
    elif sort_mode == "Bester Match (raw)":
        items.sort(key=lambda x: (safe_int(x.get("_raw_match_percent", -1), -1),
                                  safe_int(x.get("_match_percent", -1), -1),
                                  safe_int(x.get("_conf_rank", 0), 0),
                                  safe_int(x.get("_src_rank", 0), 0)), reverse=True)
    elif sort_mode == "Quelle":
        items.sort(key=lambda x: (x.get("_src_rank", 0), x.get("_match_percent", -1)), reverse=True)
    else:
        items.sort(key=lambda x: (x.get("title", "") or "").lower())

    st.caption(f"Kandidat: {search.get('name','')} | Workflow: {search.get('candidate_group_id','')}")
    st.divider()

    selected_indices: List[int] = []

    for idx, job in enumerate(items, start=1):
        title = job.get("title", "")
        src = job.get("source", "")
        link = job.get("link", "")
        snippet = job.get("snippet", "")
        company = job.get("company", "") or "Unbekannt"

        strict_mp = safe_int(job.get("_match_percent", -1), -1)
        raw_mp = safe_int(job.get("_raw_match_percent", -1), -1)
        conf = (job.get("_match", {}) or {}).get("confidence", "")
        hot_flag = bool(job.get("_hot"))

        with st.container(border=True):
            top = st.columns([0.6, 6, 1.4, 1.4])
            if bulk:
                checked = top[0].checkbox("", key=f"sel_{idx}")
                if checked:
                    selected_indices.append(idx)
            else:
                top[0].write("")

            label = f"**{title}**"
            if strict_mp >= 0 or raw_mp >= 0:
                label += f" — 🎯 strict {strict_mp}% | ⚡ raw {raw_mp}% ({conf})"
            if hot_flag:
                label += " — 🔥 HOT"
            top[1].markdown(label)
            top[1].caption(f"Quelle: {src} | Firma: {company}")

            if link:
                top[2].markdown(f"[🔗 Öffnen]({link})")
            else:
                top[2].write("")

            if top[3].button("💾 Speichern", key=f"save_one_{idx}"):
                save_job_lead_from_queue(workspace_id, search, job, workflow_order=idx)
                st.toast("Gespeichert.")
                st.rerun()

            with st.expander("Details"):
                if snippet:
                    st.write(snippet)

                m = job.get("_match") or {}
                if m:
                    st.caption(m.get("explanation", ""))
                    if m.get("why_over_90"):
                        st.success(m.get("why_over_90", ""))

                    cov = m.get("must_have_coverage", []) or []
                    if cov:
                        missing = [x for x in cov if not x.get("covered")]
                        if missing:
                            st.warning(f"Fehlende Must-Haves: {len(missing)}")
                        st.markdown("**Must-Haves Coverage**")
                        st.dataframe(pd.DataFrame(cov), use_container_width=True, hide_index=True)

                    blockers = m.get("constraint_blockers", []) or []
                    if blockers:
                        st.warning("Constraint-Blocker: " + "; ".join(blockers))

                    rf = m.get("risk_flags", []) or []
                    if rf:
                        st.warning("Risiken: " + "; ".join(rf))

    if bulk:
        st.divider()
        a, b, c = st.columns([2, 2, 2])
        if a.button("💾 Auswahl speichern", use_container_width=True, disabled=(len(selected_indices) == 0)):
            for k in selected_indices:
                job = items[k - 1]
                save_job_lead_from_queue(workspace_id, search, job, workflow_order=k)
            st.success(f"{len(selected_indices)} Jobs gespeichert.")
            st.rerun()

        if b.button("🧹 Queue leeren", use_container_width=True):
            st.session_state.pop("scan_results", None)
            st.success("Queue geleert.")
            st.rerun()

        if c.button("↩️ Zurück zu Scan", use_container_width=True):
            st.info("Wechsle im Menü zu Markt-Scan.")
            return


# =========================
# UI: Workflows
# =========================
def render_workflows(workspace_id: str):
    st.header("🧩 Workflows (Kandidat ↔ Jobs)")

    df = load_df(workspace_id)
    if df.empty:
        st.info("Keine Daten.")
        return

    groups = sorted([g for g in df["candidate_group_id"].fillna("").unique().tolist() if g.strip()])
    if not groups:
        st.info("Noch keine Workflows. Erst Screening + Jobs speichern.")
        return

    selected = st.selectbox("Workflow auswählen", groups)
    dfw = df[df["candidate_group_id"] == selected].copy()

    df_jobs = dfw[dfw["source"].fillna("") != "Screening"].copy()
    if df_jobs.empty:
        st.warning("Noch keine Jobs gespeichert. Scan → Review-Queue → speichern.")
        return

    df_jobs["strict_sort"] = pd.to_numeric(df_jobs.get("match_percent", pd.Series(dtype=float)), errors="coerce").fillna(-1)
    df_jobs["raw_sort"] = pd.to_numeric(df_jobs.get("raw_match_percent", pd.Series(dtype=float)), errors="coerce").fillna(-1)
    df_jobs = df_jobs.sort_values(["strict_sort", "raw_sort", "workflow_order", "id"], ascending=[False, False, True, False])

    st.subheader("Jobs (strict bester zuerst, dann raw)")
    show_cols = [c for c in ["status", "match_percent", "raw_match_percent", "match_confidence", "firma", "position", "source", "link"] if c in df_jobs.columns]
    st.dataframe(df_jobs[show_cols], use_container_width=True)

    st.divider()
    st.subheader("Abarbeiten (Next Best)")
    open_states = ["Hot", "Offen", "Kontaktiert", "Interview"]
    next_df = df_jobs[df_jobs["status"].isin(open_states)]
    if next_df.empty:
        st.success("Keine offenen Jobs mehr in diesem Workflow.")
        return

    row = next_df.iloc[0].to_dict()
    lead_id = safe_int(row.get("id"), 0)
    strict_mp = safe_int(row.get("match_percent"), -1)
    raw_mp = safe_int(row.get("raw_match_percent"), -1)

    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**{row.get('firma','')}** — {row.get('position','')}")
    c2.metric("Match", f"strict {strict_mp}% | raw {raw_mp}%")

    if row.get("link"):
        st.markdown(f"[🔗 Job öffnen]({row.get('link')})")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("➡️ Kontaktiert", use_container_width=True):
        update_status(workspace_id, lead_id, "Kontaktiert"); st.rerun()
    if b2.button("➡️ Interview", use_container_width=True):
        update_status(workspace_id, lead_id, "Interview"); st.rerun()
    if b3.button("✅ Placement", use_container_width=True):
        update_status(workspace_id, lead_id, "Placement"); st.rerun()
    if b4.button("⛔ Abgelehnt", use_container_width=True):
        update_status(workspace_id, lead_id, "Abgelehnt"); st.rerun()


# =========================
# UI: Kanban
# =========================
def render_kanban(workspace_id: str):
    st.header("📋 Kanban")

    df = load_df(workspace_id)
    if df.empty:
        st.info("Keine Daten.")
        return

    with st.expander("Export"):
        data = export_to_excel(df)
        filename = "Sniper.xlsx" if data[:2] == b"PK" else "Sniper.csv"
        st.download_button("Download", data, file_name=filename)

    st.divider()

    statuses = ["Screening", "Hot", "Offen", "Kontaktiert", "Interview", "Placement", "Abgelehnt"]
    cols = st.columns(len(statuses))

    for i, st_name in enumerate(statuses):
        with cols[i]:
            st.markdown(f"### {st_name}")
            sub = df[df["status"] == st_name].copy()
            sub["strict_sort"] = pd.to_numeric(sub.get("match_percent", pd.Series(dtype=float)), errors="coerce").fillna(-1)
            sub["raw_sort"] = pd.to_numeric(sub.get("raw_match_percent", pd.Series(dtype=float)), errors="coerce").fillna(-1)
            sub = sub.sort_values(["strict_sort", "raw_sort", "id"], ascending=[False, False, False])

            if sub.empty:
                st.caption("—")
                continue

            for _, row in sub.iterrows():
                lead_id = safe_int(row.get("id"), 0)
                name = row.get("kandidat_name", "") or ""
                firma = row.get("firma", "") or ""
                pos = row.get("position", "") or ""
                strict_mp = safe_int(row.get("match_percent"), -1)
                raw_mp = safe_int(row.get("raw_match_percent"), -1)

                header = f"👤 **{name}**"
                if strict_mp >= 0 or raw_mp >= 0:
                    header += f" · 🎯 {strict_mp}% | ⚡ {raw_mp}%"
                if st_name == "Hot":
                    header += " · 🔥"
                st.write(header)

                if firma and firma != "Noch offen":
                    st.caption(f"🏢 {firma} — {pos}")

                b1, b2 = st.columns(2)
                if i > 0 and b1.button("⬅️", key=f"b_{lead_id}"):
                    update_status(workspace_id, lead_id, statuses[i-1]); st.rerun()
                if i < len(statuses)-1 and b2.button("➡️", key=f"n_{lead_id}"):
                    update_status(workspace_id, lead_id, statuses[i+1]); st.rerun()

                with st.expander("Details"):
                    tab1, tab2, tab3 = st.tabs(["Profil", "Match", "Audit"])
                    with tab1:
                        st.write(f"**Quelle:** {row.get('source','')}")
                        st.write(f"**Ort:** {row.get('location','')} | **Gehalt:** {row.get('gehalt','')}")
                        st.write(f"**Skills:** {row.get('skills','')}")
                        st.text_area("Exposé", value=row.get("expose_text","") or "", height=150, key=f"exp_{lead_id}")

                        ms = safe_int(row.get("mail_sent_count"), 0)
                        ml = row.get("mail_last_sent_date","") or ""
                        st.caption(f"📧 Mail sent: {ms} | last: {ml}")
                        if st.button("✅ Mail als gesendet zählen", key=f"mail_{lead_id}"):
                            increment_mail_sent(workspace_id, lead_id)
                            st.toast("Gezählt.")
                            st.rerun()

                        pdf_data = create_candidate_pdf(row)
                        st.download_button("📥 PDF Exposé", pdf_data, file_name=f"HG_{name or lead_id}.pdf", key=f"pdf_{lead_id}")

                    with tab2:
                        m = safe_json_loads(row.get("match_json","") or "") or {}
                        if not m:
                            st.caption("Kein Match gespeichert.")
                        else:
                            st.caption(m.get("explanation", ""))
                            if m.get("why_over_90"):
                                st.success(m.get("why_over_90", ""))

                            cov = m.get("must_have_coverage", []) or []
                            if cov:
                                missing = [x for x in cov if not x.get("covered")]
                                if missing:
                                    st.warning(f"Fehlende Must-Haves: {len(missing)}")
                                st.markdown("**Must-Haves Coverage**")
                                st.dataframe(pd.DataFrame(cov), use_container_width=True, hide_index=True)

                            blockers = m.get("constraint_blockers", []) or []
                            if blockers:
                                st.warning("Constraint-Blocker: " + "; ".join(blockers))

                            rf = m.get("risk_flags", []) or []
                            if rf:
                                st.warning("Risiken: " + "; ".join(rf))

                    with tab3:
                        actions = load_actions_df(workspace_id, lead_id)
                        if actions.empty:
                            st.caption("Noch keine Aktionen geloggt.")
                        else:
                            actions2 = actions.copy()
                            actions2["meta_json"] = actions2["meta_json"].fillna("").apply(lambda s: (s[:120] + "…") if len(s) > 120 else s)
                            st.dataframe(actions2, use_container_width=True, hide_index=True)

                    if st_name == "Placement":
                        st.divider()
                        st.subheader("💰 Placement / Abrechnung")
                        with st.form(f"pl_{lead_id}"):
                            fee_total = st.number_input(
                                "Kunden-Honorar (Fee) in EUR",
                                min_value=0.0,
                                value=float(safe_float(row.get("provision"), 20000.0)),
                                step=500.0
                            )
                            currency = st.selectbox("Währung", ["EUR", "CHF", "USD"], index=0)
                            share_percent = st.number_input("Dein Anteil (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.5)
                            share_min = st.number_input("Mindestbetrag pro Placement (EUR)", min_value=0.0, value=750.0, step=50.0)
                            invoice_status = st.selectbox("Invoice Status", ["Pending", "Invoiced", "Paid"], index=0)
                            invoice_note = st.text_input("Notiz (optional)", value="")
                            submit = st.form_submit_button("Placement speichern/aktualisieren")

                        if submit:
                            upsert_placement(
                                workspace_id=workspace_id,
                                lead_row=row,
                                fee_total=float(fee_total),
                                currency=currency,
                                share_percent=float(share_percent),
                                share_min=float(share_min),
                                invoice_status=invoice_status,
                                invoice_note=invoice_note
                            )
                            st.success("Placement gespeichert.")
                            st.rerun()

                    if st.button("🗑 Löschen", key=f"del_{lead_id}", use_container_width=True):
                        delete_lead(workspace_id, lead_id); st.rerun()


# =========================
# UI: Billing
# =========================
def render_billing(workspace_id: str):
    st.header("🧾 Placements & Abrechnung")

    plc = load_placements_df(workspace_id)
    if plc.empty:
        st.info("Noch keine Placements erfasst. Im Kanban bei Status 'Placement' → Placement speichern.")
        return

    default_month = month_key(today_date())
    months = sorted(plc["placed_at"].fillna("").apply(lambda x: x[:7] if len(str(x)) >= 7 else "").unique().tolist(), reverse=True)
    if default_month not in months:
        months = [default_month] + months

    selected_month = st.selectbox("Monat", months, index=0)
    dfm = plc[plc["placed_at"].fillna("").str.startswith(selected_month)].copy()

    st.caption(f"Workspace: {workspace_id} | Monat: {selected_month}")

    total_fee = dfm["fee_total"].fillna(0).sum()
    total_share = dfm["share_amount"].fillna(0).sum()

    a, b, c = st.columns(3)
    a.metric("Placements", int(len(dfm)))
    b.metric("Fees (Summe)", f"{float(total_fee):,.2f} €")
    c.metric("Dein Anteil (Summe)", f"{float(total_share):,.2f} €")

    st.divider()
    st.subheader("Liste")
    show_cols = ["placed_at", "kandidat_name", "firma", "position", "fee_total", "share_percent", "share_min", "share_amount", "invoice_status", "invoice_note"]
    st.dataframe(dfm[show_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Export")
    data = export_to_excel(dfm[show_cols])
    filename = f"Placements_{selected_month}.xlsx" if data[:2] == b"PK" else f"Placements_{selected_month}.csv"
    st.download_button("Download Report", data, file_name=filename)


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Sniper SaaS (Match Fix + Radius)", layout="wide")
    init_db()

    st.sidebar.title("🎯 Sniper SaaS")
    nav = st.sidebar.radio(
        "Menü",
        ["Dashboard", "Screening", "Markt-Scan", "Review-Queue", "Workflows", "Kanban", "Placements & Abrechnung", "Einstellungen"]
    )

    if nav == "Einstellungen":
        render_settings()
        return

    workspace_id = get_workspace_id()

    oa_key, sa_key = get_api_keys()
    if not oa_key or not sa_key:
        st.warning("Bitte API-Keys in Einstellungen setzen oder via st.secrets/ENV hinterlegen.")
        st.info("Menü → Einstellungen")
        return

    if nav == "Dashboard":
        render_dashboard(workspace_id)
    elif nav == "Screening":
        render_screening(workspace_id, oa_key)
    elif nav == "Markt-Scan":
        render_market_scan(workspace_id, oa_key, sa_key)
    elif nav == "Review-Queue":
        render_review_queue(workspace_id)
    elif nav == "Workflows":
        render_workflows(workspace_id)
    elif nav == "Kanban":
        render_kanban(workspace_id)
    elif nav == "Placements & Abrechnung":
        render_billing(workspace_id)
    else:
        render_dashboard(workspace_id)


if __name__ == "__main__":
    main()
