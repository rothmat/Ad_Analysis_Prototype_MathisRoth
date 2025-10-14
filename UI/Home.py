# UI/Home.py
import streamlit as st
from utils_app import page_links  # nur der Link-Tree-Helper wird genutzt

st.set_page_config(page_title="Politische Werbung – Analyse", page_icon="🗳️", layout="wide")

st.title("🗳️ Politische Werbung – Analyse")
st.caption("Schnell von Rohdaten zu Insights, Kampagnen & Risiken.")

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown(
        """
**Was kann das Tool?** *(Module – identisch zur Navigation unten)*

- 🧾 **Ad Explorer:** Finde Ads in Sekunden, Themen & Strategien erkennen.
- 🖼️ **Screenshots:** Ad-Screenshots erfassen/aktualisieren, Status-Liveanzeige.
- 🧠 **LLM-Analyse:** Klassifizierung & Insights per LLM (optional).
- 📊 **Ads Overview:** Überblick, Filter & schnelle Auswertungen.
- 🧑‍🤝‍🧑 **Audience & Regionen:** Demografie & Regionen-Ansicht.
- 🎨 **Creative Insights:** Layout, Farben, Botschaften, CTAs.
- 🧭 **Kampagnen:** Erkennung, Gruppierung & Metadaten.
- 🛡️ **Schwachstellen & Risiken:** Risiko-Matrix mit Begründungen.
- 📈 **Trends & Alerts:** Themen-Shifts, Spend-Spikes & Treiber-Ads.
- 📊 **Ads Gesamtperspektive:** Aggregierte Sicht über alle Ads/Zeiträume.
- ⚙️ **Einstellungen:** Datenquellen, DB, Exporte & Defaults.
"""
    )

    if st.button("🚀 Jetzt starten", type="primary", use_container_width=True):
        try:
            st.switch_page("pages/01_Ad_Explorer.py")
        except Exception:
            st.info("Navigation nicht direkt möglich. Nutze die Schnellnavigation unten.")

with col2:
    st.info(
        "**So nutzt du das Tool**\n\n"
        "1) **Einstellungen** prüfen (Datenquelle/DB/Exporte).\n"
        "2) **Ad Explorer** öffnen und filtern.\n"
        "3) Optional **Screenshots** erfassen und **LLM-Analyse** starten.\n"
        "4) In **Detail-Tabs** tiefer springen (Overview, Audience, Creative, Kampagnen, Risiken, Trends)."
    )

st.divider()
st.subheader("Schnellnavigation")

# Link-Tree zu allen Subpages (Labels spiegeln die Liste oben)
page_links([
    ("🧾 Ad Explorer", "pages/01_Ad_Explorer.py"),
    ("🖼️ Screenshots", "pages/02_Screenshot_Capture.py"),
    ("🧠 LLM-Analyse", "pages/03_LLM_Analysis.py"),
    ("📊 Ads Overview", "pages/04_Ads_Overview.py"),
    ("🧑‍🤝‍🧑 Audience & Regionen", "pages/05_Audience_Regionen.py"),
    ("🎨 Creative Insights", "pages/06_Creative_Insights.py"),
    ("🧭 Kampagnen", "pages/07_Kampagnen.py"),
    ("🛡️ Schwachstellen & Risiken", "pages/08_Schwachstellen_Risiken.py"),
    ("📈 Trends & Alerts", "pages/09_Trends_Alerts.py"),
    ("📊 Ads Gesamtperspektive", "pages/10_Ads_Gesamtperspektive.py"),
    ("⚙️ Einstellungen", "pages/11_Einstellungen.py"),
])

with st.expander("Tipps & Hinweise", expanded=False):
    st.markdown(
        "- Du kannst jederzeit oben links zur **Home** zurückkehren.\n"
        "- Große Tabellen werden virtualisiert angezeigt (AgGrid, falls verfügbar).\n"
        "- LLM-Calls sind optional – Heuristiken liefern erste Ergebnisse.\n"
        "- UI bleibt responsiv: teure Aggregationen sind gecached."
    )
