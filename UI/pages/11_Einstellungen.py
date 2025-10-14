# pages/08_Einstellungen.py
import streamlit as st
import os
from utils_app import get_db

st.set_page_config(page_title="Einstellungen", page_icon="⚙️", layout="wide")
st.title("⚙️ Einstellungen")

st.subheader("API Keys")
st.text_input("OpenAI API Key", type="password", key="OPENAI_API_KEY", help="Wird in st.session_state gespeichert.")
st.caption("Hinweis: Für produktive Nutzung besser serverseitig per Secrets verwalten.")

st.subheader("Feature-Flags")
st.toggle("AgGrid verwenden (falls installiert)", key="flag_aggrid", value=True)
st.toggle("LLM-Ad-Tagging erlauben", key="flag_llm_adtag", value=True)

st.subheader("DB (optional)")
st.text_input("DATABASE_URL", value=os.getenv("DATABASE_URL",""), key="DATABASE_URL")
if st.button("Verbindung testen"):
    try:
        db = get_db()
        st.success("DB-Connector initialisiert (Platzhalter).")
    except Exception as e:
        st.error(f"Fehler: {e}")

st.subheader("Persistente Session")
if st.button("Session leeren"):
    st.session_state.clear()
    st.success("Session geleert.")
