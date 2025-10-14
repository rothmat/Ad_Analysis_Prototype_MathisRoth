# pages/07_Kampagnen.py
# 🎯 Kampagnen – Themen (Ad-Tagging) & Perspektiven (Kampagnen)
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# --- DB
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from agents._db import connect

# --- Externe Logik (NICHT in der UI implementiert)
import importlib
import agents.ad_tagger as adt
import agents.campaign_classifier as camp

# Hot-reload robust machen (Streamlit)
adt = importlib.reload(adt)
camp = importlib.reload(camp)

# ------------------------- Page Setup -------------------------
st.set_page_config(page_title="Kampagnen", page_icon="🎯", layout="wide")
st.title("🎯 Kampagnen – Themen & Perspektiven")

# ------------------------- DB-Verbindung ----------------------
try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")
    st.stop()

# Fallback-Tabellenerzeuger (falls Attribut in Modul beim Reload fehlt)
def _ensure_topics_table_local(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ad_topics_results (
            ad_id              INTEGER PRIMARY KEY,
            campaign_id        INTEGER,
            campaign_slug      TEXT,
            page_name          TEXT,
            bylines            TEXT[],
            media_id           TEXT,
            topics             JSONB,
            rationale_bullets  JSONB,
            confidence         DOUBLE PRECISION,
            model              TEXT,
            analyzed_at        TIMESTAMPTZ DEFAULT now()
        )""")
    conn.commit()

def _ensure_perspective_table_local(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS campaign_perspective_results (
            id                BIGSERIAL PRIMARY KEY,
            campaign_slug     TEXT NOT NULL,
            page_name         TEXT NOT NULL,
            topic             TEXT NOT NULL,
            stance            TEXT NOT NULL,
            confidence        DOUBLE PRECISION,
            rationale_bullets JSONB,
            model             TEXT,
            ad_ids            JSONB,
            analyzed_at       TIMESTAMPTZ DEFAULT now(),
            UNIQUE (campaign_slug, page_name, topic)
        )""")
    conn.commit()

# Robust aufrufen (verhindert AttributeError bei teilinitialisierten Modulen)
getattr(adt,  "ensure_topics_table",      _ensure_topics_table_local)(conn)
getattr(camp, "ensure_perspective_table", _ensure_perspective_table_local)(conn)

# ------------------------- Auswahl Kampagne -------------------
camps = adt.list_campaigns(conn)  # [(id, name, slug)]
if not camps:
    st.info("Keine Kampagnen in der DB gefunden.")
    st.stop()

label_for = lambda row: f"{row[1]} ({row[2]})"
idx = st.selectbox("Kampagne", options=list(range(len(camps))),
                   format_func=lambda i: label_for(camps[i]))
campaign_id, campaign_name, campaign_slug = camps[idx]

# ------------------------- Ads aus ad_llm_fused ----------------
# Wichtig: Alle Ads aus ad_llm_fused anzeigen, NICHT nur bereits getaggte.
ads_df = adt.list_fused_ads_for_campaign(conn, campaign_slug)

# ❌ Entfernt: Begrenzung auf bereits analysierte Ads (hat Auswahl eingeschränkt)
# analyzed_ids = set(adt.list_analyzed_ad_ids(conn, campaign_slug))
# ads_df = ads_df[ads_df["ad_pk"].isin(analyzed_ids)].copy()

if ads_df.empty:
    st.info("Für diese Kampagne sind (noch) keine Ads in ad_llm_fused vorhanden.")
    st.stop()

options  = ads_df["label"].tolist()
label2pk = dict(zip(ads_df["label"], ads_df["ad_pk"]))

SEL_KEY   = "__sel_ads__"
ALL_FLAG  = "__sel_ads_all_flag__"

# ---- 1) „Alle auswählen“ Flag abarbeiten (MUSS vor dem Widget passieren!)
if st.session_state.get(ALL_FLAG):
    st.session_state[SEL_KEY] = options[:]   # jetzt ist das Widget noch nicht gerendert ⇒ erlaubt
    st.session_state.pop(ALL_FLAG, None)

# ---- 2) UI
col_sel, col_btn = st.columns([5, 1])
with col_sel:
    prev = st.session_state.get(SEL_KEY, options[:min(8, len(options))])
    default_vals = [v for v in prev if v in options] or options[:min(8, len(options))]
    sel_labels = st.multiselect(
        "Ads auswählen (nur aus ad_llm_fused)",
        options=options,
        default=default_vals,
        key=SEL_KEY,   # Widget ist an Session gebunden
    )
with col_btn:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    if st.button("Alle auswählen", use_container_width=True):
        st.session_state[ALL_FLAG] = True
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

selected_ad_pks = [label2pk[l] for l in st.session_state.get(SEL_KEY, []) if l in label2pk]
if not selected_ad_pks:
    st.info("Bitte mindestens eine Ad auswählen.")
    st.stop()

# ------------------------- Filter (Gruppe/Sponsor) ------------
st.subheader("Filter")
c1, c2 = st.columns(2)

with c1:
    use_group = st.checkbox("Gruppe filtern", value=False, key="fltr_group")
    group_opts = sorted(ads_df["page_name"].dropna().astype(str).unique().tolist())
    sel_groups = st.multiselect("Gruppe (page_name)", options=group_opts, disabled=not use_group)

with c2:
    use_sponsor = st.checkbox("Sponsoren filtern", value=False, key="fltr_sponsor")
    all_bylines = sorted(set(sum(ads_df["bylines"].tolist(), [])))
    sel_bylines = st.multiselect("Sponsoren (bylines)", options=all_bylines, disabled=not use_sponsor)

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if use_group and sel_groups:
        out = out[out["page_name"].isin(set(sel_groups))]
    if use_sponsor and sel_bylines:
        out = out[out["bylines"].apply(lambda bl: bool(set(bl) & set(sel_bylines)))]
    return out

# ------------------------- LLM-Buttons (auf der Page) ---------
api_key = os.getenv("OPENAI_API_KEY") or ""
if not api_key:
    st.warning("OPENAI_API_KEY nicht gesetzt – bitte in der Umgebung hinterlegen (.env oder System-Env).")

col_a, col_b, col_force = st.columns([1.5, 1.8, 1.0])
with col_force:
    force = st.toggle("Erneut analysieren", value=False, help="Existierende DB-Einträge überschreiben.")

def _progress_hook(label: str):
    text = st.empty()
    bar  = st.progress(0.0)

    def _cb(i: int, n: int | float, stage: str):
        # n robust normalisieren
        try:
            n_int = int(n)
        except Exception:
            n_int = 0

        if n_int <= 0:
            frac = 0.0
        else:
            frac = (i + 1) / n_int
            # in [0,1] clampen
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

        text.write(f"{label}: **{stage}** – {i+1}/{max(1, n_int)}")
        bar.progress(frac)

    return _cb

with col_a:
    if st.button("🔎 Themen (Ad-Tagging) starten", use_container_width=True, disabled=not api_key):
        cb = _progress_hook("Ad-Tagging")
        adt.tag_ads_to_db(
            conn=conn,
            campaign_slug=campaign_slug,
            ad_pks=selected_ad_pks,
            api_key=api_key,
            model="gpt-4o-mini",
            force=force,
            progress_cb=cb,
        )
        st.success("Ad-Tagging abgeschlossen.")

with col_b:
    if st.button("🧭 Perspektiven (Kampagnen) starten", use_container_width=True, disabled=not api_key):
        cb = _progress_hook("Kampagnen-Perspektiven")
        camp.classify_perspective_to_db(
            conn=conn,
            campaign_slug=campaign_slug,
            ad_pks=selected_ad_pks,
            api_key=api_key,
            model="gpt-4o-mini",
            force=force,
            progress_cb=cb,
        )
        st.success("Kampagnen-Perspektiven abgeschlossen.")

st.divider()

# ------------------------- Ergebnisse: Ad-Themen --------------
st.header("Ergebnisse – Themen (Ad-Tagging)")

topics_df = adt.fetch_topics_results(
    conn=conn,
    campaign_slug=campaign_slug,
    ad_pks=selected_ad_pks
)

if topics_df.empty:
    st.caption("Keine (gefilterten) Ergebnisse vorhanden.")
else:
    enrich = ads_df[["ad_pk", "label", "page_name", "bylines"]].copy()
    topics_df = topics_df.merge(
        enrich, on="ad_pk", how="left", suffixes=("", "_ads")
    )
    if "page_name_ads" in topics_df.columns:
        topics_df["page_name"] = topics_df["page_name"].fillna(topics_df["page_name_ads"])
        topics_df.drop(columns=["page_name_ads"], inplace=True, errors="ignore")
    if "bylines_ads" in topics_df.columns:
        topics_df["bylines"] = topics_df["bylines"].where(
            topics_df["bylines"].notna(), topics_df["bylines_ads"]
        )
        topics_df.drop(columns=["bylines_ads"], inplace=True, errors="ignore")

    topics_df = _apply_filters(topics_df)

    if topics_df.empty:
        st.caption("Keine (gefilterten) Ergebnisse vorhanden.")
    else:
        pretty = topics_df.rename(columns={
            "label": "Ad",
            "topics": "Themen",
            "confidence": "Konfidenz",
            "rationale_bullets": "Begründungen",
        })
        cols_wanted = ["Ad", "page_name", "Themen", "Konfidenz", "Begründungen"]
        cols_show = [c for c in cols_wanted if c in pretty.columns]
        st.dataframe(pretty[cols_show], use_container_width=True)

        exp = topics_df.explode("topics")
        if not exp.empty and "topics" in exp.columns:
            vc = (
                exp["topics"]
                .value_counts(dropna=False)
                .rename_axis("Thema")
                .reset_index(name="Anzahl")
            )
            st.plotly_chart(
                px.bar(vc, x="Thema", y="Anzahl", title="Themenhäufigkeit (ausgewählte Ads)"),
                use_container_width=True
            )

# ------------------------- Ergebnisse: Perspektiven -----------
st.header("Ergebnisse – Perspektiven (Actor × Topic)")

persp_df = camp.fetch_perspective_results(conn=conn, campaign_slug=campaign_slug)
actors_df = ads_df[["page_name"]].drop_duplicates()

if persp_df.empty:
    st.caption("Keine (gefilterten) Perspektiv-Ergebnisse vorhanden.")
else:
    persp_df = actors_df.merge(persp_df, on="page_name", how="left")
    persp_df = _apply_filters(persp_df)

    if persp_df.dropna(subset=["topic"]).empty:
        st.caption("Keine (gefilterten) Perspektiv-Ergebnisse vorhanden.")
    else:
        pretty = persp_df.rename(columns={
            "page_name": "Akteur (Gruppe)",
            "topic": "Thema",
            "stance": "Perspektive",
            "confidence": "Konfidenz",
            "rationale_bullets": "Begründungen"
        })
        st.dataframe(
            pretty[["Akteur (Gruppe)", "Thema", "Perspektive", "Konfidenz", "Begründungen"]],
            use_container_width=True
        )

        score_map = {"Pro": 1.0, "Contra": -1.0, "Neutral": 0.0, "Unklar": 0.0}
        tmp = persp_df.copy()
        tmp["score"] = tmp["stance"].map(score_map).fillna(0.0)

        pivot_mean = (
            tmp.pivot_table(index="page_name", columns="topic",
                            values="score", aggfunc="mean", fill_value=0.0)
            .sort_index(axis=0).sort_index(axis=1)
        )
        pivot_cnt = (
            tmp.pivot_table(index="page_name", columns="topic",
                            values="score", aggfunc="count", fill_value=0)
            .reindex_like(pivot_mean).fillna(0).astype(int)
        )
        pro_share = (
            (tmp.assign(is_pro=(tmp["stance"] == "Pro").astype(int)))
            .pivot_table(index="page_name", columns="topic",
                         values="is_pro", aggfunc="mean", fill_value=0.0)
            .reindex_like(pivot_mean).fillna(0.0)
        )
        contra_share = (
            (tmp.assign(is_contra=(tmp["stance"] == "Contra").astype(int)))
            .pivot_table(index="page_name", columns="topic",
                         values="is_contra", aggfunc="mean", fill_value=0.0)
            .reindex_like(pivot_mean).fillna(0.0)
        )

        fig = px.imshow(
            pivot_mean,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            origin="upper",
            aspect="auto",
        )
        fig.update_layout(
            title=f"Haltung zum Thema – bezogen auf Kampagne „{campaign_name}“ (−1 = Gegen, 0 = Neutral/Unklar, +1 = Für)",
            xaxis_title="Thema",
            yaxis_title="Akteur (Gruppe)",
            coloraxis_colorbar=dict(
                title="Für ↔ Gegen (Kampagnenkontext)",
                tickvals=[-1, 0, 1],
                ticktext=["Gegen", "Neutral/Unklar", "Für"],
            ),
            margin=dict(l=10, r=10, t=70, b=10),
        )

        cd = np.dstack([
            pivot_cnt.values,
            pro_share.values,
            contra_share.values,
        ])
        fig.update_traces(
            customdata=cd,
            hovertemplate=(
                "Akteur: %{y}<br>"
                "Thema: %{x}<br>"
                "Balance: %{z:.2f} (−1=Gegen, +1=Für)<br>"
                "Beiträge: %{customdata[0]:.0f}<br>"
                "Pro-Anteil: %{customdata[1]:.0%}<br>"
                "Contra-Anteil: %{customdata[2]:.0%}<extra></extra>"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Definition: „Für“ = befürwortet Thema/Policy, „Gegen“ = lehnt ab – jeweils im Kontext der ausgewählten Kampagne.")
