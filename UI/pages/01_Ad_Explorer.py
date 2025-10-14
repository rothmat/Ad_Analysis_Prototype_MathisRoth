# pages/01_Ad_Explorer.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils_app import aggrid_or_dataframe, nav_to, get_query_ad, set_query_ad, ad_detail_view

# API-Funktionen
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agents"))
from meta_api_agent import run_search, generate_keywords, flatten_results  # type: ignore

st.set_page_config(page_title="Ad Explorer", page_icon="🧾", layout="wide")
st.title("🧾 Ad Explorer")

with st.expander("🔎 Ads abrufen (Meta Ad Library)", expanded=True):
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        topic = st.text_input("Thema / Kampagne", placeholder="z. B. Eigenmietwert")
    with c2:
        country = st.selectbox("Region (Country)", ["", "CH", "DE", "FR", "IT", "EU"], index=1)
    with c3:
        per_lang = st.number_input("Suchbegriffe je Sprache", min_value=10, max_value=10, value=10, step=1,
                                   help="Fix: 10 je Sprache")

    # Zeitraum
    today = date.today()
    default_from = today - timedelta(days=30)
    daterange = st.date_input("Zeitraum", (default_from, today), format="YYYY-MM-DD")

    # Keyword-Vorschau (persistiert)
    kw_state_key = "kw_preview"
    if kw_state_key not in st.session_state:
        st.session_state[kw_state_key] = None

    cA, cB = st.columns([1, 1])
    with cA:
        gen_disabled = not bool(topic)
        if st.button("💡 Keywords generieren", disabled=gen_disabled):
            with st.status("Erzeuge Keywords …", expanded=True) as status:
                kws = generate_keywords(topic, per_lang=10)
                st.session_state[kw_state_key] = kws
                status.update(label="Keywords bereit.", state="complete")

    with cB:
        valid_dates = isinstance(daterange, tuple) and len(daterange) == 2
        have_kws = isinstance(st.session_state.get(kw_state_key), dict)
        disabled = not (topic and country and valid_dates and have_kws)
        start = st.button("▶️ Abruf starten", type="primary", disabled=disabled)

    # Vorschau der Keywords
    kws = st.session_state.get(kw_state_key)
    if kws:
        st.caption("Verwendete Suchbegriffe (je 10)")
        col_de, col_fr, col_it = st.columns(3)
        with col_de: st.write("**Deutsch**");      st.dataframe(pd.DataFrame({"de": kws.get("de", [])}), use_container_width=True)
        with col_fr: st.write("**Französisch**");  st.dataframe(pd.DataFrame({"fr": kws.get("fr", [])}), use_container_width=True)
        with col_it: st.write("**Italienisch**");  st.dataframe(pd.DataFrame({"it": kws.get("it", [])}), use_container_width=True)

    # Ergebnisse
    result_key = "last_fetch_results"
    if start:
        with st.status("Starte Abruf …", expanded=True) as status:
            def _cb(msg, p): status.update(label=msg, state="running")
            d_from, d_to = daterange[0], daterange[1]
            out = run_search(
                topic=topic,
                country=country or "CH",
                date_from=d_from, date_to=d_to,
                per_lang=10,
                write_db=True,
                progress=_cb,
                forced_keywords=st.session_state[kw_state_key],
            )
            st.session_state[result_key] = out
            status.update(label=f"Fertig – {len(out['results'])} Ads", state="complete")

out = st.session_state.get("last_fetch_results")
if out:
    st.subheader("Ergebnisse (vollständige Tabelle)")
    df_all = pd.DataFrame(out["results"])
    try:
        df_all = flatten_results(df_all)
    except Exception:
        pass
    st.dataframe(df_all, use_container_width=True, height=420)

    st.caption(f"Datei: `{out['saved_file']}`  ·  Zeitraum: {out['date_from']} → {out['date_to']}  ·  Region: {out['country']}")
    if out["db"].get("written"):
        st.success(f"DB: {out['db']['inserted_ads']} Ads gespeichert (Snapshot {out['db'].get('snapshot_date')}).")
    elif out["db"].get("error"):
        st.warning(f"DB: nicht geschrieben – {out['db']['error']}")

# # --- dein bestehender Explorer darunter (ohne unteren 'Ad auswählen'-Dropdown) ---


# df = st.session_state.get("raw_import_df")
# if df is None or df.empty:
#     st.info("Keine Daten geladen. Gehe zu **Merge & Export** und importiere Daten.")
#     if st.button("Zum Import"): nav_to("pages/07_Merge_Export.py")
#     st.stop()

# # Filterzeile
# q = st.text_input("Suchen (ID, Text, Kampagne, Akteur)", key="ad_search")
# colf1, colf2, colf3 = st.columns(3)
# with colf1:
#     actor = st.selectbox("Akteur", ["— alle —"] + sorted(df["advertiser"].dropna().unique().tolist()))
# with colf2:
#     plats = sorted(set(sum([str(p).split(", ") for p in df["platforms"].fillna("").tolist()], [])))
#     plat = st.selectbox("Plattform", ["— alle —"] + plats)
# with colf3:
#     rng = st.date_input("Zeitraum", value=(pd.to_datetime(df["start_date"]).min(), pd.to_datetime(df["end_date"]).max()))

# dff = df.copy()
# if actor != "— alle —": dff = dff[dff["advertiser"] == actor]
# if plat != "— alle —": dff = dff[dff["platforms"].fillna("").str.contains(plat)]
# if isinstance(rng, tuple) and len(rng)==2:
#     start_d, end_d = pd.to_datetime(rng[0]).date(), pd.to_datetime(rng[1]).date()
#     dff = dff[(pd.to_datetime(dff["end_date"]).dt.date >= start_d) & (pd.to_datetime(dff["start_date"]).dt.date <= end_d)]
# if q:
#     ql = q.lower()
#     dff = dff[
#         dff["id"].astype(str).str.contains(q)
#         | dff["campaign"].fillna("").str.lower().str.contains(ql)
#         | dff["advertiser"].fillna("").str.lower().str.contains(ql)
#         | dff["ad_text"].fillna("").str.lower().str.contains(ql)
#     ]

# c_list, c_detail = st.columns([1.6, 1.4])
# with c_list:
#     st.caption(f"{len(dff):,} Treffer".replace(",", "."))
#     show = dff[["id","advertiser","campaign","platforms","start_date","end_date","spend_mid","impressions_mid"]].copy()
#     show["id"] = show["id"].astype(str)
#     aggrid_or_dataframe(show, height=500, key="ad_explorer_grid")

# with c_detail:
#     st.subheader("Details")
#     sel_id = st.session_state.get("selected_ad_id") or get_query_ad()
#     if sel_id: ad_detail_view(sel_id)
#     else: st.info("Bitte links eine Ad auswählen.")