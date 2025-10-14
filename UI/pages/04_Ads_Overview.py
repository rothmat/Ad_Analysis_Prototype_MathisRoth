# pages/04_Ads_Overview.py
# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import streamlit as st

# DB-Client laden (wie in den anderen Seiten)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

st.set_page_config(page_title="Ads Overview", page_icon="🗂️", layout="wide")
st.title("🗂️ Ads Overview")

# ------------------------------- DB Helpers -------------------------------

def list_campaigns(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, slug FROM campaigns ORDER BY name")
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "name", "slug"])


def list_ads_for_campaign(conn, slug: str) -> pd.DataFrame:
    """
    Liefert NUR Ads der Kampagne, die mind. einen Eintrag in ad_llm_fused haben
    (inner LATERAL join auf den neuesten fused + zusätzliches EXISTS als Guard).
    Der jüngste fused liefert uns auch gleich den page_name fürs Label.
    """
    sql = """
      SELECT
        a.id                         AS ad_id,
        a.ad_external_id             AS ad_external_id,
        (ff.fused->'api'->'raw'->>'page_name') AS page_name
      FROM ads a
      JOIN campaigns c ON c.id = a.campaign_id
      -- nur Ads mit mind. einem fused
      JOIN LATERAL (
          SELECT fused
          FROM ad_llm_fused f
          WHERE f.ad_id = a.id
          ORDER BY f.snapshot_date DESC, f.created_at DESC
          LIMIT 1
      ) ff ON TRUE
      WHERE c.slug = %s
        AND EXISTS (SELECT 1 FROM ad_llm_fused f2 WHERE f2.ad_id = a.id)
      ORDER BY a.id DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (slug,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def fused_rows_for_ad(conn, ad_id: int) -> pd.DataFrame:
    sql = """
      SELECT id AS fused_id, snapshot_date, fused, created_at
      FROM ad_llm_fused
      WHERE ad_id = %s
      ORDER BY snapshot_date DESC, created_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ad_id,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def _parse_media_id_from_llm_payload(fused_obj: dict) -> Optional[int]:
    try:
        llm = (fused_obj or {}).get("llm_analysis") or {}
        payload = llm.get("analysis_file_payload") or {}
        sid = payload.get("screenshot_id") or ""
        if isinstance(sid, str) and sid.startswith("media:"):
            return int(sid.split(":",1)[1])
    except Exception:
        pass
    return None


def _fetch_b64_by_media_id(conn, media_id: int) -> Tuple[Optional[str], Optional[str], Optional[pd.Timestamp]]:
    sql = """
      SELECT mb.b64, mb.mime_type, m.created_at
      FROM media m
      LEFT JOIN media_base64 mb ON mb.media_id = m.id
      WHERE m.id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (media_id,))
        row = cur.fetchone()
    if not row:
        return None, None, None
    b64, mime, ts = row
    return b64, mime, pd.to_datetime(ts) if ts else None


def _fetch_latest_screenshot_for_ad(conn, ad_id: int) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[pd.Timestamp]]:
    sql = """
      SELECT m.id, mb.b64, mb.mime_type, m.created_at
      FROM media m
      LEFT JOIN media_base64 mb ON mb.media_id = m.id
      WHERE m.ad_id = %s AND m.kind = 'screenshot'
      ORDER BY m.created_at DESC
      LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ad_id,))
        row = cur.fetchone()
    if not row:
        return None, None, None, None
    mid, b64, mime, ts = row
    return int(mid), b64, mime, pd.to_datetime(ts) if ts else None


def fetch_screenshot_for_fused(conn, ad_id: int, fused_obj: dict) -> Tuple[Optional[int], Optional[str], Optional[str], str]:
    """
    Priorität:
      1) media_id aus LLM-Payload (screenshot_id="media:<id>")
      2) erster Screenshot in fused.media.screenshots
      3) neuester Screenshot der Ad
    """
    # 1) aus LLM
    mid = _parse_media_id_from_llm_payload(fused_obj)
    if mid:
        b64, mime, ts = _fetch_b64_by_media_id(conn, mid)
        if b64 and mime:
            cap = f"media_id: {mid} · {ts:%Y-%m-%d %H:%M}" if ts is not None else f"media_id: {mid}"
            return mid, b64, mime, cap

    # 2) aus fused.media
    try:
        media_list = ((fused_obj or {}).get("media") or {}).get("screenshots") or []
        if media_list:
            mid2 = int(media_list[0].get("media_id"))
            b64, mime, ts = _fetch_b64_by_media_id(conn, mid2)
            if b64 and mime:
                cap = f"media_id: {mid2} · {ts:%Y-%m-%d %H:%M}" if ts is not None else f"media_id: {mid2}"
                return mid2, b64, mime, cap
    except Exception:
        pass

    # 3) Fallback: neuester Screenshot der Ad
    mid3, b64, mime, ts = _fetch_latest_screenshot_for_ad(conn, ad_id)
    if b64 and mime:
        cap = f"media_id: {mid3} · {ts:%Y-%m-%d %H:%M}" if ts is not None else f"media_id: {mid3}"
        return mid3, b64, mime, cap

    return None, None, None, "Kein Screenshot verfügbar"

# --------------------------- Aufbereitung / UI ----------------------------

def _mid_range(d: dict, lo_key="lower_bound", hi_key="upper_bound") -> Optional[float]:
    try:
        lo = float((d or {}).get(lo_key, "nan"))
        hi = float((d or {}).get(hi_key, "nan"))
        if pd.isna(lo) or pd.isna(hi):
            return None
        return (lo + hi) / 2.0
    except Exception:
        return None

def _fmt_span(lo: Any, hi: Any) -> str:
    lo_s = "-" if lo in (None, "") else str(lo)
    hi_s = "-" if hi in (None, "") else str(hi)
    return f"{lo_s} – {hi_s}"

def render_api_summary(raw: dict) -> None:
    spend = raw.get("spend") or {}
    impr  = raw.get("impressions") or {}
    page  = raw.get("page_name") or raw.get("page_name_en") or raw.get("page_name_de") or raw.get("page_id")
    start = raw.get("ad_delivery_start_time") or raw.get("ad_creation_time")
    stop  = raw.get("ad_delivery_stop_time")
    currency = (raw.get("currency") or "CHF").upper()

    spend_mid = _mid_range(spend)
    impr_mid  = _mid_range(impr)

    def fmt_money(val: Optional[float]) -> str:
        if val is None:
            return "–"
        return f"{val:.0f} {currency}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Spend (Mittel)", fmt_money(spend_mid))
        st.caption(f"Range: {_fmt_span(spend.get('lower_bound'), spend.get('upper_bound'))} {currency}")
    with c2:
        st.metric("Impressions (Mittel)", f"{impr_mid:.0f}" if impr_mid is not None else "–")
        st.caption(f"Range: {_fmt_span(impr.get('lower_bound'), impr.get('upper_bound'))}")
    with c3:
        st.metric("Page", str(page) if page else "–")
        st.caption(f"Laufzeit: {str(start)[:10]} → {str(stop)[:10] if stop else 'offen'}")

    # Demografie (kurzer Überblick)
    demo = raw.get("demographic_distribution") or []
    if demo:
        st.subheader("Demografie (Top 10)")
        demo_rows = []
        for d in demo:
            demo_rows.append({
                "Gender": (d.get("gender") or "unknown").title(),
                "Age": d.get("age") or "unknown",
                "Share": float(d.get("percentage") or 0.0)
            })
        ddf = pd.DataFrame(demo_rows).sort_values("Share", ascending=False).head(10)
        st.dataframe(ddf, use_container_width=True)

    # Regionen (kurzer Überblick)
    regs = raw.get("region_distribution") or raw.get("delivery_by_region") or []
    if regs:
        st.subheader("Regionen (Top 10)")
        rows = []
        for r in regs:
            region = r.get("region") or r.get("name") or r.get("key") or "Unbekannt"
            perc = r.get("percentage") or r.get("share") or r.get("value") or 0
            try:
                perc = float(str(perc).replace("%",""))
            except Exception:
                perc = 0.0
            rows.append({"Region": region, "Share": perc})
        rr = pd.DataFrame(rows)
        st.dataframe(rr.sort_values("Share", ascending=False).head(10), use_container_width=True)

    # Creatives aus API
    bodies = raw.get("ad_creative_bodies") or raw.get("ad_creative_body") or []
    if isinstance(bodies, str):
        bodies = [bodies]
    if bodies:
        st.subheader("Ad-Text (API)")
        for i, t in enumerate(bodies[:3], start=1):
            st.write(f"**Text {i}:** {t}")

def render_llm_summary(payload: dict) -> None:
    geeignet = payload.get("geeignet", True)
    analyse  = payload.get("analyse") or {}

    st.markdown(f"**Geeignet:** {'Ja' if geeignet else 'Nein'}")
    vis = analyse.get("visuelle_features") or {}
    txt = analyse.get("textuelle_features") or {}
    sem = analyse.get("semantische_features") or {}

    c1, c2, c3 = st.columns(3)
    with c1:
        pal = vis.get("farbpalette") or []
        st.metric("Farben erkannt", str(len(pal)))
        st.caption(", ".join([str(x) for x in pal[:5]]))
    with c2:
        st.metric("CTA-Typ", txt.get("cta_typ", "Unklar"))
        st.caption(f"Prominenz: {txt.get('cta_visuelle_prominenz','Unklar')}, Pos.: {txt.get('cta_position','–')}")
    with c3:
        st.metric("Plattform", vis.get("plattform","Unbekannt"))
        st.caption(f"Layout: {vis.get('dominante_layoutstruktur','–')}")

    fl = (vis.get("flächenverteilung") or {})
    if fl:
        st.subheader("Flächenverteilung (%)")
        st.dataframe(pd.DataFrame([{
            "Text": fl.get("textfläche"),
            "Bild": fl.get("bildfläche"),
            "Weißraum": fl.get("weißraum")
        }]), use_container_width=True)

    if sem:
        st.subheader("Semantik – Kurzüberblick")
        bullets = [
            f"- **Werbeversprechen:** {sem.get('werbeversprechen','–')}",
            f"- **Zielgruppe (Modell):** {sem.get('zielgruppe','–')}",
            f"- **Ansprache:** {sem.get('ansprache_typ','–')}",
            f"- **Framing:** {sem.get('framing_typ','–')}",
            f"- **Emotionaler Appell:** {sem.get('emotionaler_apell','–')}"
        ]
        st.markdown("\n".join(bullets))

# --------------------------------- UI Flow ---------------------------------

# 1) DB verbinden
try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")
    st.stop()

# 2) Kampagne wählen
camps = list_campaigns(conn)
if camps.empty:
    st.info("Keine Kampagnen vorhanden."); st.stop()

colA, colB = st.columns([2,3])
with colA:
    camp_sel = st.selectbox(
        "Kampagne",
        options=camps["slug"],
        format_func=lambda s: f"{camps.set_index('slug').loc[s,'name']} ({s})"
    )

# 3) Ads laden & wählen (Label: <Gruppe/Akteur> (<ad_external_id>), nur mit fused)
ads_df = list_ads_for_campaign(conn, camp_sel)
if ads_df.empty:
    st.info("Für diese Kampagne wurden noch keine **fused** Ads erfasst.")
    st.stop()

def _label_row(r):
    page = (r.get("page_name") or "").strip() or "Unbekannte Gruppe"
    ext  = r.get("ad_external_id")
    return f"{page} ({ext})"

ads_df["label"] = ads_df.apply(_label_row, axis=1)

with colB:
    idx = st.selectbox(
        "Ad auswählen",
        options=ads_df.index,
        format_func=lambda i: ads_df.loc[i, "label"]
    )
sel_ad_id = int(ads_df.loc[idx, "ad_id"])

# 4) Fused-Rows für Ad
fused_list = fused_rows_for_ad(conn, sel_ad_id)
if fused_list.empty:
    st.warning("Für diese Ad existieren noch keine Einträge in ad_llm_fused.")
    st.stop()

# Falls mehrere Snapshots existieren → auswählbar
snapshots = fused_list["snapshot_date"].dropna().unique().tolist()
snapshots_sorted = sorted([s for s in snapshots if s is not None], reverse=True)
snap_sel = st.selectbox("Snapshot-Tag", options=snapshots_sorted, index=0, format_func=lambda d: d.isoformat())

fused_row = fused_list[fused_list["snapshot_date"] == snap_sel].iloc[0]
fused_obj = fused_row["fused"]
if isinstance(fused_obj, str):
    try:
        fused_obj = json.loads(fused_obj)
    except Exception:
        fused_obj = {}

# 5) Screenshot besorgen
mid, b64, mime, caption = fetch_screenshot_for_fused(conn, sel_ad_id, fused_obj)

# 6) Layout: Screenshot & Details
colL, colR = st.columns([1,1])

with colL:
    st.subheader("Screenshot")
    if b64 and mime:
        st.image(f"data:{mime};base64,{b64}", caption=caption, use_container_width=True)
    else:
        st.info("Kein Screenshot verfügbar.")

with colR:
    # API Block
    api_raw = ((fused_obj or {}).get("api") or {}).get("raw") or {}
    st.subheader("API – Zusammenfassung")
    if api_raw:
        render_api_summary(api_raw)
    else:
        st.caption("Keine API-Daten im Fused-Objekt.")

# LLM Block (unter beiden Spalten)
st.divider()
st.subheader("LLM – Zusammenfassung")
llm_payload = ((fused_obj or {}).get("llm_analysis") or {}).get("analysis_file_payload") or {}
if llm_payload:
    render_llm_summary(llm_payload)
else:
    st.caption("Keine LLM-Analyse vorhanden.")

# Rohdaten-Expander
with st.expander("API – Raw JSON"):
    st.json(api_raw if api_raw else {})
with st.expander("LLM – Raw JSON (payload)"):
    st.json(llm_payload if llm_payload else {})
