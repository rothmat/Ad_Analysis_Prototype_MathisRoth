#06_Creative_Insights
# -*- coding: utf-8 -*-
import json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

st.set_page_config(page_title="Creative Insights", page_icon="🎨", layout="wide")
st.title("🎨 Creative Insights")

# ---------------- DB helpers ----------------

def get_campaign_options(conn) -> List[Tuple[int, str, str]]:
    sql = "SELECT id, name, slug FROM campaigns ORDER BY name"
    with conn.cursor() as cur:
        cur.execute(sql)
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

def get_llm_ads_for_campaign(conn, campaign_slug: str) -> pd.DataFrame:
    """
    Holt alle Ads einer Kampagne, die LLM/Fusion haben.
    Liefert ad_pk und ad_external_id (page_name/bylines etc. kommen aus fused.api.raw).
    """
    sql = """
      SELECT DISTINCT a.id AS ad_pk, a.ad_external_id
      FROM ad_llm_fused f
      JOIN ads a ON a.id = f.ad_id
      JOIN campaigns c ON c.id = a.campaign_id
      WHERE c.slug = %s
      ORDER BY ad_pk DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug,))
        rows = cur.fetchall()
    return pd.DataFrame([{"ad_pk": int(r[0]), "ad_external_id": r[1]} for r in rows])

def load_fused_rows(conn, campaign_slug: str, ad_pks: List[int]) -> pd.DataFrame:
    if not ad_pks:
        return pd.DataFrame(columns=["ad_pk","snapshot_date","fused","ad_external_id"])
    sql = """
      SELECT f.ad_id AS ad_pk,
             f.snapshot_date,
             f.fused,
             f.created_at,
             a.ad_external_id
      FROM ad_llm_fused f
      JOIN ads a ON a.id = f.ad_id
      JOIN campaigns c ON c.id = a.campaign_id
      WHERE c.slug = %s AND f.ad_id = ANY(%s)
      ORDER BY f.snapshot_date, f.created_at
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug, ad_pks))
        rows = cur.fetchall()
    out = []
    for ad_pk, snapshot_date, fused, _created, ad_external_id in rows:
        if isinstance(fused, str):
            try:
                fused = json.loads(fused)
            except Exception:
                fused = {}
        out.append({
            "ad_pk": int(ad_pk),
            "snapshot_date": str(snapshot_date),
            "fused": fused,
            "ad_external_id": ad_external_id
        })
    return pd.DataFrame(out)

def extract_llm_blocks(fused_row: Dict[str,Any]) -> Dict[str,Any]:
    llm = ((fused_row or {}).get("llm_analysis") or {}).get("analysis_file_payload") or {}
    vis = (llm.get("analyse") or {}).get("visuelle_features") or {}
    txt = (llm.get("analyse") or {}).get("textuelle_features") or {}
    sem = (llm.get("analyse") or {}).get("semantische_features") or {}
    return {"vis": vis, "txt": txt, "sem": sem}

def _from_api_raw(fused_row: Dict[str,Any], key: str):
    try:
        return ((fused_row or {}).get("api") or {}).get("raw", {}).get(key)
    except Exception:
        return None

def _norm_list(v):
    if v is None: return []
    if isinstance(v, str): return [v]
    if isinstance(v, (list, tuple)): return [str(x) for x in v if str(x).strip()]
    return []

# ---------------- Auswahl Kampagne & Ads ----------------

try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")
    st.stop()

camp_opts = get_campaign_options(conn)
if not camp_opts:
    st.info("Keine Kampagnen gefunden."); st.stop()

labels = [f"{n} ({s})" for _, n, s in camp_opts]
idx = st.selectbox("Kampagne", options=list(range(len(camp_opts))), format_func=lambda i: labels[i])
campaign_id, campaign_name, campaign_slug = camp_opts[idx]

# 1) Kandidaten holen
ads_df = get_llm_ads_for_campaign(conn, campaign_slug)

if ads_df.empty:
    st.info("Keine Ads mit LLM/Fusion gefunden.")
    st.stop()

# 2) Alle Fused-Zeilen laden und pro Ad den jüngsten Snapshot wählen
all_pks = ads_df["ad_pk"].astype(int).tolist()
fused_all = load_fused_rows(conn, campaign_slug, all_pks)
if fused_all.empty:
    st.info("Keine fused-Daten für diese Kampagne.")
    st.stop()

tmp = fused_all.copy()
tmp["snapshot_date"] = pd.to_datetime(tmp["snapshot_date"], errors="coerce")
last_all = tmp.sort_values("snapshot_date").groupby("ad_pk", as_index=False, observed=False).tail(1)

# 3) Anzeige-Optionen bauen: "Gruppe (media_id/ad_external_id)"
def _label_for_row(r) -> str:
    pn = str(_from_api_raw(r["fused"], "page_name") or "").strip() or "Unbekannte Gruppe"
    media_id = str(_from_api_raw(r["fused"], "media_id") or r.get("ad_external_id") or "").strip()
    return f"{pn} ({media_id})" if media_id else pn

last_all = last_all.assign(
    page_name=last_all["fused"].apply(lambda f: str(_from_api_raw(f, "page_name") or "").strip() or "Unbekannte Gruppe"),
    media_id=last_all["fused"].apply(lambda f: str(_from_api_raw(f, "media_id") or "").strip()),
    bylines=last_all["fused"].apply(lambda f: _norm_list(_from_api_raw(f, "bylines"))),
)
last_all["ad_label"] = last_all.apply(_label_for_row, axis=1)

options = last_all["ad_label"].tolist()
label_to_pk = dict(zip(last_all["ad_label"], last_all["ad_pk"]))

# 4) Multi-Select + „Alle auswählen“ (wie in 07_Kampagnen.py)
SEL_KEY  = "__sel_ads__"
ALL_FLAG = "__sel_ads_all_flag__"
MAX_DEFAULT = 12

# 4.1 „Alle auswählen“-Flag vor dem Rendern abarbeiten
if st.session_state.get(ALL_FLAG):
    st.session_state[SEL_KEY] = options[:]  # alle Optionen übernehmen
    st.session_state.pop(ALL_FLAG, None)

# 4.2 UI: Multiselect + Button in zwei Spalten
col_sel, col_btn = st.columns([5, 1])
with col_sel:
    prev = st.session_state.get(SEL_KEY, options[:min(MAX_DEFAULT, len(options))])
    default_vals = [v for v in prev if v in options] or options[:min(MAX_DEFAULT, len(options))]

    sel_labels = st.multiselect(
        "Ads auswählen (nur mit LLM/Fusion)",
        options=options,
        default=default_vals,
        key=SEL_KEY,  # Session-gebunden, wie in 07_Kampagnen.py
    )

with col_btn:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    if st.button("Alle auswählen", use_container_width=True):
        st.session_state[ALL_FLAG] = True
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# 4.3 Auswahl nach ad_pk abbilden
selected_ad_pks = [label_to_pk[l] for l in st.session_state.get(SEL_KEY, []) if l in label_to_pk]

st.divider()

if not selected_ad_pks:
    st.info("Bitte mindestens eine Ad auswählen.")
    st.stop()

# 5) Gefilterte fused-Daten (nur gewählte Ads)
fused_df = last_all[last_all["ad_pk"].isin(selected_ad_pks)].copy()

# ---------------- Optionale Filter (wie Audience & Regionen) ----------------
st.subheader("Filter")
c1, c2 = st.columns(2)

with c1:
    use_group = st.checkbox("Gruppe filtern", value=False, key="ci_filter_group")
    group_options = sorted([g for g in fused_df["page_name"].dropna().unique().tolist() if str(g).strip()])
    sel_groups = st.multiselect("Gruppe (page_name)", options=group_options, disabled=not use_group)

with c2:
    use_sponsor = st.checkbox("Sponsoren filtern", value=False, key="ci_filter_sponsor")
    all_bylines = sorted(set(sum([bl for bl in fused_df["bylines"].tolist()], [])))
    sel_bylines = st.multiselect("Sponsoren (bylines)", options=all_bylines, disabled=not use_sponsor)

def _passes_filters(row) -> bool:
    ok_g = True
    ok_b = True
    if use_group:
        ok_g = row.get("page_name") in set(sel_groups) if sel_groups else True
    if use_sponsor:
        rb = set(row.get("bylines") or [])
        ok_b = bool(rb & set(sel_bylines)) if sel_bylines else True
    return bool(ok_g and ok_b)

fused_df["__keep"] = fused_df.apply(_passes_filters, axis=1)
fused_df = fused_df.loc[fused_df["__keep"]].drop(columns="__keep")

if fused_df.empty:
    st.info("Nach Filtern keine Ads übrig."); st.stop()

# ---- Creative-Features extrahieren (nur gefilterte Ads)
rows=[]
for row in fused_df.itertuples():
    blocks = extract_llm_blocks(row.fused)
    rows.append({
        "ad_pk": row.ad_pk,
        "ad_label": row.ad_label,
        "page_name": row.page_name,
        "bylines": row.bylines,
        "vis": blocks["vis"],
        "txt": blocks["txt"],
        "sem": blocks["sem"]
    })
cre = pd.DataFrame(rows)

if cre.empty:
    st.caption("Keine Creative-Features gefunden.")
    st.stop()

# ---------------- Flächenverteilung (stacked bars)
stack=[]
for r in cre.itertuples():
    fl = (r.vis or {}).get("flächenverteilung") or {}
    def _num(x):
        try:
            return float(x)
        except Exception:
            return None
    t = _num(fl.get("textfläche"))
    b = _num(fl.get("bildfläche"))
    w = _num(fl.get("weißraum"))
    if any(v is not None for v in (t,b,w)):
        stack.append({"ad_label": r.ad_label, "Typ": "Text", "Anteil": t})
        stack.append({"ad_label": r.ad_label, "Typ": "Bild", "Anteil": b})
        stack.append({"ad_label": r.ad_label, "Typ": "Weißraum", "Anteil": w})

st.subheader("Flächenverteilung (%)")
if stack:
    sdf = pd.DataFrame(stack).dropna()
    fig_area = px.bar(
        sdf, x="ad_label", y="Anteil", color="Typ", barmode="stack",
        title="Text/Bild/Weißraum je Ad"
    )
    fig_area.update_layout(xaxis_title=None, yaxis_title="Anteil",
                           xaxis_tickangle=-20, margin=dict(l=10,r=10,b=60,t=40))
    st.plotly_chart(fig_area, use_container_width=True)
else:
    st.caption("Keine Flächenverteilung gefunden.")

# --- CTA-Typen & Prominenz
cta=[]
for r in cre.itertuples():
    t = r.txt or {}
    cta.append({"CTA": t.get("cta_typ","Unklar"),
                "Prominenz": (t.get("cta_visuelle_prominenz") or "Unklar")})
ctadf = pd.DataFrame(cta)
if not ctadf.empty:
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("CTA-Typen")
        plot_df = ctadf.groupby("CTA", as_index=False).size().rename(columns={"size":"count"})
        st.plotly_chart(px.bar(plot_df, x="CTA", y="count", title=None),
                        use_container_width=True)
    with c2:
        st.subheader("CTA-Prominenz")
        plot_df = ctadf.groupby("Prominenz", as_index=False).size().rename(columns={"size":"count"})
        st.plotly_chart(px.bar(plot_df, x="Prominenz", y="count", title=None),
                        use_container_width=True)

# --- Layout/Komposition/Plattform
dist=[]
for r in cre.itertuples():
    vis = r.vis or {}
    dist.append({"Komposition": vis.get("kompositionstyp","?"),
                 "Layout": vis.get("dominante_layoutstruktur","?"),
                 "Plattform": vis.get("plattform","Unbekannt")})
ddf = pd.DataFrame(dist)
c1,c2,c3 = st.columns(3)
with c1:
    st.subheader("Komposition")
    plot_df = ddf.groupby("Komposition", as_index=False).size().rename(columns={"size":"count"})
    st.plotly_chart(px.bar(plot_df, x="Komposition", y="count", title=None),
                    use_container_width=True)
with c2:
    st.subheader("Layout")
    plot_df = ddf.groupby("Layout", as_index=False).size().rename(columns={"size":"count"})
    st.plotly_chart(px.bar(plot_df, x="Layout", y="count", title=None),
                    use_container_width=True)
with c3:
    st.subheader("Plattform (LLM)")
    plot_df = ddf.groupby("Plattform", as_index=False).size().rename(columns={"size":"count"})
    st.plotly_chart(px.bar(plot_df, x="Plattform", y="count", title=None),
                    use_container_width=True)

# ---------------- Farbpalette – Abdeckung je Farbe (über ausgewählte Ads)
def _norm_hex(c: str) -> Optional[str]:
    if not isinstance(c, str): return None
    s = c.strip()
    if not s: return None
    if not s.startswith("#"): return None
    s = s.upper()
    if len(s) == 4 and all(ch in "0123456789ABCDEF" for ch in s[1:]):
        s = "#" + "".join(ch*2 for ch in s[1:])
    return s

colors_by_ad = {}
for r in cre.itertuples():
    pal = (r.vis or {}).get("farbpalette") or []
    if isinstance(pal, list):
        normed = {c for c in (_norm_hex(str(x)) for x in pal) if c}
        if normed:
            colors_by_ad[r.ad_pk] = normed

n_ads = len(cre["ad_pk"].unique())
coverage = {}
for _ad, cols in colors_by_ad.items():
    for c in cols:
        coverage[c] = coverage.get(c, 0) + 1

cov_rows = []
for c, cnt in sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0])):
    pct = 0.0 if n_ads == 0 else (cnt / n_ads) * 100.0
    cov_rows.append({"Farbe": c, "Abdeckung (%)": pct})
cov_df = pd.DataFrame(cov_rows)

st.subheader("Farbpalette – Abdeckung (Anteil der Ads mit der Farbe)")
if cov_df.empty:
    st.caption("Keine Farbwerte gefunden.")
else:
    cov_df = cov_df.sort_values("Abdeckung (%)", ascending=True)
    fig_cols = px.bar(
        cov_df,
        x="Abdeckung (%)", y="Farbe",
        orientation="h",
        text=cov_df["Abdeckung (%)"].map(lambda v: f"{v:.0f}%"),
    )
    fig_cols.update_traces(marker=dict(color=cov_df["Farbe"].tolist()),
                           textposition="outside", cliponaxis=False)
    fig_cols.update_layout(
        xaxis_range=[0, 100],
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(260, 26*len(cov_df) + 120)
    )
    st.plotly_chart(fig_cols, use_container_width=True)

# --- Textmetriken KPIs
tlen = []
for r in cre.itertuples():
    t = r.txt or {}
    def _num(x):
        if x in (None, "Unklar"): return math.nan
        try: return float(x)
        except: return math.nan
    tlen.append([_num(t.get("headline_zeichenanzahl")),
                 _num(t.get("anzahl_textblöcke")),
                 _num(t.get("durchschnittliche_wortlänge"))])
if tlen:
    arr = np.array(tlen, dtype=float)
    m1 = float(np.nanmean(arr[:,0])) if arr.size else math.nan
    m2 = float(np.nanmean(arr[:,1])) if arr.size else math.nan
    m3 = float(np.nanmean(arr[:,2])) if arr.size else math.nan
    k1,k2,k3 = st.columns(3)
    k1.metric("Ø Headline-Zeichen", f"{m1:.0f}" if not math.isnan(m1) else "–")
    k2.metric("Ø Textblöcke", f"{m2:.1f}" if not math.isnan(m2) else "–")
    k3.metric("Ø Wortlänge", f"{m3:.1f}" if not math.isnan(m3) else "–")
