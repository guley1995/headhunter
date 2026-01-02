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
    # Wir sagen der KI SEHR deutlich, wie das JSON aussehen muss
    prompt = (
        "Analysiere den Lebenslauf. Erstelle ein JSON mit exakt diesen Schlüsseln: "
        "'name', 'skills', 'wohnort', 'search_query'. "
        "Der 'search_query' muss ein Suchbegriff für Google Jobs sein (z.B. 'Projektleiter Jobs München')."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# --- LOGIK BEI KLICK (Verbessert) ---
if st.button("🚀 Analyse starten"):
    if not (openai_key and serpapi_key and uploaded_file):
        st.error("Bitte alle Felder ausfüllen!")
    else:
        with st.spinner("Sniper analysiert den Lebenslauf..."):
            text = extract_text_from_pdf(uploaded_file)
            profile = analyze_cv(text, openai_key)
            
            # FEHLER-CHECK: Falls 'search_query' fehlt, bauen wir ihn manuell
            query = profile.get('search_query')
            if not query:
                # Fallback: Jobtitel (falls da) + Ort
                query = f"Jobs {profile.get('wohnort', 'Deutschland')}"
            
            st.info(f"Suche gestartet für: {query}")
            
            jobs = search_jobs(query, serpapi_key)
            
            if not jobs:
                st.warning("Keine Jobs gefunden. Versuche es mit einem anderen Suchbegriff.")
            else:
                # Hier geht es weiter mit dem Matching...
                for job in jobs[:8]:
                    m = match_job(job, profile, openai_key)
                    # ... restlicher Code wie gehabt

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
