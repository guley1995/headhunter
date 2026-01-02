import streamlit as st
import requests
import json
import pdfplumber
import openai

# --- UI SETUP ---
st.set_page_config(page_title="Reverse Recruiting Sniper", layout="wide")

st.title("🎯 Reverse Recruiting Sniper")
st.markdown("Lade deinen Lebenslauf hoch und finde automatisch perfekt passende Jobs inkl. Sales-Mail.")

# Sidebar für API-Keys
with st.sidebar:
    st.header("🔑 API Konfiguration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    serpapi_key = st.text_input("SerpApi Key", type="password")
    st.info("Keys werden nur für die aktuelle Sitzung genutzt.")

# Datei-Uploader
uploaded_file = st.file_uploader("Lebenslauf hochladen (PDF)", type="pdf")

# --- FUNKTIONEN ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def analyze_cv(text, key):
    client = openai.OpenAI(api_key=key)
    prompt = "Extrahiere aus dem CV: Name, Skills, Wohnort. Erstelle einen Google-Jobs Suchbegriff (Titel Jobs Ort). Gib NUR JSON zurück."
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def search_jobs(query, key):
    url = "https://serpapi.com/search.json"
    params = {"engine": "google_jobs", "q": query, "hl": "de", "gl": "de", "api_key": key}
    return requests.get(url, params=params).json().get("jobs_results", [])

def match_job(job, profile, key):
    client = openai.OpenAI(api_key=key)
    prompt = "Vergleiche Job und Kandidat. Score 0-100. Schreibe aggressive Sales-Mail. Gib NUR JSON zurück: {match_score, status, mail_subject, mail_body}"
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Profil: {profile}\nJob: {job}"}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# --- LOGIK BEI KLICK ---
if st.button("🚀 Analyse starten"):
    if not (openai_key and serpapi_key and uploaded_file):
        st.error("Bitte alle Felder ausfüllen!")
    else:
        with st.spinner("Sniper arbeitet..."):
            text = extract_text_from_pdf(uploaded_file)
            profile = analyze_cv(text, openai_key)
            jobs = search_jobs(profile['search_query'], serpapi_key)
            
            for job in jobs[:8]: # Top 8 Jobs prüfen
                m = match_job(job, profile, openai_key)
                if m['match_score'] >= 85:
                    with st.expander(f"⭐ {m['match_score']}% - {job.get('company_name')}"):
                        st.subheader(m['mail_subject'])
                        st.code(m['mail_body'], language="text")
                        st.link_button("Job öffnen", job.get("related_links", [{}])[0].get("link", "#"))
