import streamlit as st
import requests
import json
import pdfplumber
import openai
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Sniper V8.1 - HG Mediadesign Edition", layout="wide")

def extract_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except: return ""

def main():
    st.title("🎯 Sniper V8.1 – HG Mediadesign Agency Edition")

    # --- SIDEBAR: MANUELLE EINTRAGUNGEN ---
    with st.sidebar:
        st.header("🔑 API Konfiguration")
        oa_key = st.text_input("OpenAI Key", type="password")
        sa_key = st.text_input("SerpApi Key", type="password")
        
        st.divider()
        st.header("👤 Absender-Details")
        # Hier kannst du jetzt alles manuell ändern/eintragen
        rec_name = st.text_input("Dein Name", value="Hüsnü Güley")
        rec_agency = st.text_input("Agentur", value="HG Mediadesign")
        rec_mail = st.text_input("E-Mail", value="office@hg-mediadesign.de")
        
        st.divider()
        st.header("⚙️ Pitch-Optionen")
        lang = st.radio("Sprache", ["Deutsch", "English"])
        style = st.select_slider("Stil", options=["Dezent", "Professionell", "Vertriebsstark"], value="Professionell")
        
        if st.button("🗑️ Cache Reset"):
            st.session_state.clear()
            st.rerun()

    # --- MAIN UI ---
    uploaded_file = st.file_uploader("Kandidaten-CV (PDF) hochladen", type="pdf")

    if uploaded_file:
        # Analyse-Logik
        if "cv_data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Erstelle Markt-Strategie..."):
                text = extract_text(uploaded_file)
                if not oa_key:
                    st.warning("Bitte OpenAI Key eingeben!")
                    return
                
                client = openai.OpenAI(api_key=oa_key)
                prompt = (
                    "Analysiere diesen CV. Erstelle ein anonymisiertes Exposé (max 400 Zeichen). "
                    "Gib 3 Suchbegriffe (spezifisch bis allgemein). "
                    "Antworte als JSON: {'expose': '...', 'location': '...', 'queries': ['...', '...', '...']}"
                )
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
                        response_format={"type": "json_object"}
                    )
                    st.session_state.cv_data = json.loads(resp.choices[0].message.content)
                    st.session_state.file_name = uploaded_file.name
                except Exception as e:
                    st.error(f"KI Fehler: {e}")

        if "cv_data" in st.session_state:
            data = st.session_state.cv_data
            
            with st.expander("📄 Anonymisiertes Exposé Vorschau"):
                st.write(data['expose'])

            st.subheader("🔍 Markt-Scan")
            c1, c2 = st.columns(2)
            # Manuelle Anpassung der Suche (wie in V7)
            query = c1.text_input("Suchbegriff", value=data['queries'][1])
            location = c2.text_input("Region", value=data['location'])

            if st.button("🚀 Markt jetzt besnipern", use_container_width=True, type="primary"):
                run_v8_engine(query, location, sa_key, oa_key, data, lang, style, rec_name, rec_agency, rec_mail)

def run_v8_engine(q, l, sa_key, oa_key, cv_data, lang, style, r_name, r_agency, r_mail):
    client = openai.OpenAI(api_key=oa_key)
    
    with st.status(f"Suche nach '{q}'...", expanded=True) as status:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google_jobs", "q": q, "location": l, "hl": "de", "api_key": sa_key}
        
        try:
            r = requests.get(url, params=params, timeout=20).json()
            jobs = r.get("jobs_results", [])
            if not jobs:
                params.pop("location", None)
                r = requests.get(url, params=params).json()
                jobs = r.get("jobs_results", [])
        except:
            st.error("Suche fehlgeschlagen.")
            return

    if jobs:
        crm_list = []
        st.success(f"{len(jobs)} Firmen gefunden!")
        
        for i, job in enumerate(jobs[:8]):
            with st.expander(f"🏢 {job.get('company_name')} | {job.get('title')}"):
                
                p_prompt = (
                    f"Schreibe als Headhunter ({r_name}, {r_agency}) zwei Akquise-Pitches auf {lang}. "
                    f"Stil: {style}. Kandidat: {cv_data['expose']}. Job: {job.get('title')}. "
                    "Antworte als JSON: {'mail_sub': '...', 'mail_body': '...', 'li_msg': '...'}"
                )
                
                try:
                    p_resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": p_prompt}],
                        response_format={"type": "json_object"}
                    )
                    p = json.loads(p_resp.choices[0].message.content)
                    
                    t1, t2, t3 = st.tabs(["📧 E-Mail", "💬 LinkedIn", "📝 Exposé"])
                    with t1:
                        st.subheader(p['mail_sub'])
                        st.code(f"{p['mail_body']}\n\nBeste Grüße,\n{r_name}\n{r_agency}\n{r_mail}", language="text")
                    with t2:
                        st.code(p['li_msg'], language="text")
                    with t3:
                        st.write(cv_data['expose'])
                    
                    st.link_button("Zur Anzeige", job.get("link") or "#")
                    crm_list.append({"Firma": job.get('company_name'), "Position": job.get('title'), "Link": job.get('link')})
                except: continue
        
        if crm_list:
            st.divider()
            df = pd.DataFrame(crm_list)
            st.download_button("📊 CRM Liste (CSV) exportieren", df.to_csv(index=False).encode('utf-8'), "leads.csv", "text/csv")
    else:
        st.error("Keine Jobs gefunden. Suchbegriff im Textfeld oben einkürzen!")

if __name__ == "__main__":
    main()
