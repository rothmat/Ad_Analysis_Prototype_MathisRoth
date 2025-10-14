# pages/10_Ads_Gesamtperspektive.py
# -*- coding: utf-8 -*-
import os, json, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date as _date

# --- optionales OpenAI (nur für 3–5 Sätze Kurzbeschreibung)
try:
    from openai import OpenAI
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False

# --- DB
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

st.set_page_config(page_title="Gesamtperspektive", page_icon="🧭", layout="wide")
st.title("🧭 Gesamtperspektive")

# -------------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------------
def _parse_media_id_from_llm_payload(fused_obj: dict) -> Optional[int]:
    try:
        llm = (fused_obj or {}).get("llm_analysis") or {}
        payload = llm.get("analysis_file_payload") or {}
        sid = payload.get("screenshot_id") or ""
        if isinstance(sid, str) and sid.startswith("media:"):
            return int(sid.split(":", 1)[1])
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
    mid = _parse_media_id_from_llm_payload(fused_obj)
    if mid:
        b64, mime, ts = _fetch_b64_by_media_id(conn, mid)
        if b64 and mime:
            cap = "Screenshot" + (f" · {ts:%Y-%m-%d %H:%M}" if ts is not None else "")
            return mid, b64, mime, cap
    try:
        media_list = ((fused_obj or {}).get("media") or {}).get("screenshots") or []
        if media_list:
            mid2 = int(media_list[0].get("media_id"))
            b64, mime, ts = _fetch_b64_by_media_id(conn, mid2)
            if b64 and mime:
                cap = f"media_id: {mid2}" + (f" · {ts:%Y-%m-%d %H:%M}" if ts is not None else "")
                return mid2, b64, mime, cap
    except Exception:
        pass
    mid3, b64, mime, ts = _fetch_latest_screenshot_for_ad(conn, ad_id)
    if b64 and mime:
        cap = f"media_id: {mid3}" + (f" · {ts:%Y-%m-%d %H:%M}" if ts is not None else "")
        return mid3, b64, mime, cap
    return None, None, None, "Kein Screenshot verfügbar"

def _norm_list(v):
    if v is None: return []
    if isinstance(v, str): return [v]
    if isinstance(v, (list, tuple)): return [str(x) for x in v if str(x).strip()]
    return []

def _num(x) -> float:
    try:
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"): return float(s[:-1]) / 100.0
            return float(s)
        return float(x)
    except Exception:
        return float("nan")

import re

def _mid_value(v):
    """
    Liefert einen numerischen Schätzwert für spend/impressions aus verschiedensten Formaten:
    - dict mit lower_bound/upper_bound (Strings oder Zahlen)
    - Zahl (int/float)
    - String-Buckets wie '<1000', '1K-5K', '10k-50k', '50k-100k', '100k-200k', '200k-500k', '>1M'
    """
    if v is None:
        return 0.0

    # 1) Dict-Range
    if isinstance(v, dict):
        lb = v.get("lower_bound")
        ub = v.get("upper_bound")
        try:
            lb_f = float(str(lb).replace(",", "").replace("K","000").replace("M","000000")) if lb is not None else None
            ub_f = float(str(ub).replace(",", "").replace("K","000").replace("M","000000")) if ub is not None else None
            if lb_f is not None and ub_f is not None:
                return (lb_f + ub_f) / 2.0
            if lb_f is not None and ub_f is None:
                return float(lb_f)
            if lb_f is None and ub_f is not None:
                return float(ub_f)
        except Exception:
            pass
        return 0.0

    # 2) Reine Zahl
    if isinstance(v, (int, float)):
        return float(v)

    # 3) String-Buckets
    if isinstance(v, str):
        s = v.strip().upper().replace(" ", "")
        # '<1000' → 500
        if s.startswith("<"):
            try:
                num = float(s[1:].replace("K","000").replace("M","000000"))
                return max(num/2.0, 0.0)
            except Exception:
                return 0.0
        # '>1M' → konservativ ~1.25M (Mittel zwischen 1M und 1.5M)
        if s.startswith(">"):
            try:
                num = float(s[1:].replace("K","000").replace("M","000000"))
                # nimm 1.25 * Schwelle als Faustwert
                return num * 1.25
            except Exception:
                return 0.0
        # '1K-5K', '10K-50K', '1000-1999' etc.
        m = re.match(r"^([0-9.,KkMm]+)-([0-9.,KkMm]+)$", s)
        if m:
            a, b = m.groups()
            try:
                a_f = float(a.replace(",", "").replace("K","000").replace("M","000000"))
                b_f = float(b.replace(",", "").replace("K","000").replace("M","000000"))
                return (a_f + b_f) / 2.0
            except Exception:
                return 0.0
        # reine Zahl als String
        try:
            return float(s.replace(",", "").replace("K","000").replace("M","000000"))
        except Exception:
            return 0.0

    return 0.0

# ---------------- DB helpers ----------------
def list_fully_analyzed_ads(conn, campaign_slugs: Optional[List[str]] = None) -> pd.DataFrame:
    base_sql = """
      SELECT
          a.id                    AS ad_id,
          a.ad_external_id        AS ad_external_id,
          c.id                    AS campaign_id,
          c.slug                  AS campaign_slug,
          c.name                  AS campaign_name,
          (ff.fused->'api'->'raw'->>'page_name')   AS page_name,
          (ff.fused->'api'->'raw'->'bylines')      AS bylines_json,
          (ff.fused->'api'->'raw'->>'ad_snapshot_url') AS snapshot_url
      FROM ads a
      JOIN campaigns c ON c.id = a.campaign_id
      JOIN LATERAL (
          SELECT fused
          FROM ad_llm_fused f
          WHERE f.ad_id = a.id
          ORDER BY f.snapshot_date DESC, f.created_at DESC
          LIMIT 1
      ) ff ON TRUE
      WHERE EXISTS (SELECT 1 FROM ad_topics_results t WHERE t.ad_id = a.id)
        AND EXISTS (SELECT 1 FROM ad_weaknesses w WHERE w.ad_id = a.id)
    """
    params: Tuple[Any, ...] = ()
    if campaign_slugs:
        base_sql += " AND c.slug = ANY(%s)"
        params = (list(campaign_slugs),)

    with conn.cursor() as cur:
        cur.execute(base_sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

    df = pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(
        columns=["ad_id","ad_external_id","campaign_id","campaign_slug","campaign_name",
                 "page_name","bylines_json","snapshot_url"]
    )

    def _bylines_to_list(v):
        if v is None: return []
        if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
        try:
            j = json.loads(v)
            if isinstance(j, list): return [str(x).strip() for x in j if str(x).strip()]
        except Exception:
            pass
        return []

    df["bylines"] = df["bylines_json"].apply(_bylines_to_list)
    df.drop(columns=["bylines_json"], inplace=True, errors="ignore")

    def _media_id_from_url(u: str) -> Optional[str]:
        if not isinstance(u, str): return None
        m = re.search(r"[?&]id=(\d+)", u)
        return m.group(1) if m else None

    df["media_id"] = df["snapshot_url"].apply(_media_id_from_url)

    def _label(r):
        pn  = (r.get("page_name") or "").strip() or "Unbekannte Gruppe"
        ext = r.get("ad_external_id")
        num = str(ext).strip() if ext is not None and str(ext).strip() else str(r.get("ad_id"))
        return f"{pn} ({num})"


    if not df.empty:
        df["label"] = df.apply(_label, axis=1)

    return df

def fused_history_for_ad(conn, ad_id: int) -> pd.DataFrame:
    sql = """
      SELECT snapshot_date, fused, created_at
      FROM ad_llm_fused
      WHERE ad_id = %s
      ORDER BY snapshot_date, created_at
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ad_id,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    if df.empty: return df
    def _parse(x):
        if isinstance(x, str):
            try: return json.loads(x)
            except: return {}
        return x or {}
    df["fused"] = df["fused"].apply(_parse)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df

def _ad_campaign_id(conn, ad_id: int) -> Optional[int]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT campaign_id FROM ads WHERE id=%s", (ad_id,))
            r = cur.fetchone()
        return int(r[0]) if r and r[0] is not None else None
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return None

def api_history_for_ad_external(conn, ad_id: int) -> pd.DataFrame:
    """
    Liest Zeitreihe ausschließlich aus api_snapshots:
    - filtert nach der Kampagne der Ad (campaign_id),
    - sucht in payload (Liste von Ads) nach passender ad_external_id,
    - extrahiert Midpoints von spend / impressions pro snapshot_date.
    """
    ext_id = _ad_external_id(conn, ad_id)
    camp_id = _ad_campaign_id(conn, ad_id)
    if not ext_id or not camp_id:
        return pd.DataFrame(columns=["snapshot_date","spend","impressions"])

    cols = _table_cols(conn, "api_snapshots")
    if not cols or "snapshot_date" not in cols or "payload" not in cols:
        return pd.DataFrame(columns=["snapshot_date","spend","impressions"])

    sql = """
        SELECT snapshot_date, payload, id
        FROM api_snapshots
        WHERE campaign_id = %s
        ORDER BY snapshot_date, id
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (camp_id,))
            rows = cur.fetchall()
            hdrs = [d[0] for d in cur.description]
        if not rows:
            return pd.DataFrame(columns=["snapshot_date","spend","impressions"])
        df = pd.DataFrame([dict(zip(hdrs, r)) for r in rows])
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return pd.DataFrame(columns=["snapshot_date","spend","impressions"])

    out = []
    for r in df.itertuples():
        raw = r.payload
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []

        if isinstance(raw, list):
            hit = next((x for x in raw if str(x.get("id")) == str(ext_id)), None)
            if hit:
                spend_mid = _mid_value(hit.get("spend"))
                impr_mid  = _mid_value(hit.get("impressions"))
                out.append({
                    "snapshot_date": pd.to_datetime(r.snapshot_date).date(),
                    "spend": float(spend_mid or 0.0),
                    "impressions": float(impr_mid or 0.0),
                })

    ts = pd.DataFrame(out)
    if ts.empty:
        return ts
    return (ts.groupby("snapshot_date", as_index=False)[["spend","impressions"]]
              .sum()
              .sort_values("snapshot_date"))

# --- Hilfen für robusten Fallback auf api_snapshots -----------------
def _table_cols(conn, table: str) -> List[str]:
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name=%s
            """, (table,))
            rows = cur.fetchall()
        return [r[0] for r in rows]
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return []

def _ad_external_id(conn, ad_id: int) -> Optional[str]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ad_external_id FROM ads WHERE id=%s", (ad_id,))
            r = cur.fetchone()
        return str(r[0]) if r and r[0] is not None else None
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return None

def api_history_for_ad(conn, ad_id: int) -> pd.DataFrame:
    cols = _table_cols(conn, "api_snapshots")
    if not cols:
        return pd.DataFrame(columns=["snapshot_date","raw","created_at"])

    # ad-Referenzspalte ermitteln
    where_col = None
    where_val = None
    if "ad_id" in cols:
        where_col, where_val = "ad_id", ad_id
        where_val_txt = str(ad_id)
    elif "ad_pk" in cols:
        where_col, where_val = "ad_pk", ad_id
        where_val_txt = str(ad_id)
    elif "ad_external_id" in cols:
        ext = _ad_external_id(conn, ad_id)
        if not ext:
            return pd.DataFrame(columns=["snapshot_date","raw","created_at"])
        where_col, where_val = "ad_external_id", ext
        where_val_txt = str(ext)
    else:
        return pd.DataFrame(columns=["snapshot_date","raw","created_at"])

    # wahrscheinliche JSON-/Zeitspalten
    json_candidates = ["payload", "raw", "api_raw", "snapshot"]
    ts_candidates   = ["snapshot_ts", "created_at", "ingested_at", "fetched_at", "updated_at"]
    date_col = "snapshot_date" if "snapshot_date" in cols else None
    json_col = next((c for c in json_candidates if c in cols), None)
    ts_col   = next((c for c in ts_candidates   if c in cols), None)
    if not json_col or not ts_col:
        return pd.DataFrame(columns=["snapshot_date","raw","created_at"])

    # Datum robust bauen
    date_expr = "COALESCE(snapshot_date::date, DATE({ts}))" if date_col else "DATE({ts})"
    date_expr = date_expr.format(ts=ts_col)

    sql = f"""
        SELECT {date_expr} AS snapshot_date, {json_col} AS raw, {ts_col} AS created_at
        FROM api_snapshots
        WHERE ({where_col} = %s OR {where_col}::text = %s)
        ORDER BY 1, {ts_col}
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (where_val, where_val_txt))
            rows = cur.fetchall()
            hdrs = [d[0] for d in cur.description]
        if not rows:
            return pd.DataFrame(columns=["snapshot_date","raw","created_at"])
        df = pd.DataFrame([dict(zip(hdrs, r)) for r in rows])
        # JSON parsen
        def _parse(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except: return {}
            return x or {}
        df["raw"] = df["raw"].apply(_parse)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df
    except Exception:
        try: conn.rollback()
        except Exception: pass
        return pd.DataFrame(columns=["snapshot_date","raw","created_at"])

def fetch_topics_for_ad(conn, ad_id: int) -> Optional[dict]:
    sql = """
      SELECT topics, rationale_bullets, confidence, model, analyzed_at
      FROM ad_topics_results
      WHERE ad_id = %s
      ORDER BY analyzed_at DESC
      LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ad_id,))
        row = cur.fetchone()
    if not row: return None
    topics, rationales, conf, model, ts = row
    return {
        "topics": topics or [],
        "rationale_bullets": rationales or [],
        "confidence": conf,
        "model": model,
        "analyzed_at": ts,
    }

def fetch_weakness_for_ad(conn, ad_id: int, model: Optional[str]=None) -> Optional[dict]:
    if model:
        sql = """
          SELECT model, result_json, overall_risk, overall_confidence, updated_at
          FROM ad_weaknesses
          WHERE ad_id = %s AND model = %s
          ORDER BY updated_at DESC NULLS LAST
          LIMIT 1
        """
        params = (ad_id, model)
    else:
        sql = """
          SELECT model, result_json, overall_risk, overall_confidence, updated_at
          FROM ad_weaknesses
          WHERE ad_id = %s
          ORDER BY updated_at DESC NULLS LAST
          LIMIT 1
        """
        params = (ad_id,)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    if not row: return None
    mdl, res_json, risk, conf, ts = row
    if isinstance(res_json, str):
        try: res_json = json.loads(res_json)
        except: res_json = {}
    return {
        "model": mdl, "updated_at": ts,
        "overall_risk": risk, "overall_confidence": conf,
        "payload": res_json or {}
    }

def extract_demography(api_raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Nutzt demographic_distribution (percentage, age, gender) und verteilt
    die Midpoint-Spend anteilig (spend_mid * percentage).
    """
    if not isinstance(api_raw, dict):
        return pd.DataFrame()

    dlist = api_raw.get("demographic_distribution") or []
    if not isinstance(dlist, (list, tuple)) or not dlist:
        return pd.DataFrame()

    # Midpoint aus raw['spend'] (Range oder Zahl)
    spend_mid = _mid_value(api_raw.get("spend")) or 0.0

    rows = []
    for rec in dlist:
        if not isinstance(rec, dict):
            continue
        pct = _mid_value(rec.get("percentage")) or 0.0  # "0.123456" → 0.123456
        rows.append({
            "gender": str(rec.get("gender") or "").strip(),
            "age":    str(rec.get("age") or "").strip(),
            "percentage": pct,
            "spend":  spend_mid * pct,
        })

    df = pd.DataFrame(rows)
    return df

def extract_regions(api_raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Nutzt delivery_by_region (percentage, region) und verteilt die Midpoint-Spend anteilig.
    """
    if not isinstance(api_raw, dict):
        return pd.DataFrame()

    rlist = api_raw.get("delivery_by_region") or []
    if not isinstance(rlist, (list, tuple)) or not rlist:
        return pd.DataFrame()

    spend_mid = _mid_value(api_raw.get("spend")) or 0.0

    rows = []
    for rec in rlist:
        if not isinstance(rec, dict):
            continue
        pct = _mid_value(rec.get("percentage")) or 0.0
        rows.append({
            "region": str(rec.get("region") or "").strip(),
            "percentage": pct,
            "spend": spend_mid * pct,
        })

    return pd.DataFrame(rows)

def extract_llm_blocks(fused_row: Dict[str,Any]) -> Dict[str,Any]:
    llm = ((fused_row or {}).get("llm_analysis") or {}).get("analysis_file_payload") or {}
    vis = (llm.get("analyse") or {}).get("visuelle_features") or {}
    txt = (llm.get("analyse") or {}).get("textuelle_features") or {}
    sem = (llm.get("analyse") or {}).get("semantische_features") or {}
    return {"vis": vis, "txt": txt, "sem": sem, "raw": llm}

def _norm_hex(c: str) -> Optional[str]:
    if not isinstance(c, str): return None
    s = c.strip().upper()
    if not s.startswith("#"): return None
    if len(s) == 4 and all(ch in "0123456789ABCDEF" for ch in s[1:]):
        s = "#" + "".join(ch*2 for ch in s[1:])
    return s

# ---------- Perspektive: robustes Auslesen aus verschiedenen Tabellen/Views
def get_perspectives_for_ads(conn, ad_ids: List[int]) -> pd.DataFrame:
    """
    Liefert DataFrame mit Spalten: ad_id, perspective, confidence?, rationale?
    Probiert mehrere potenzielle Tabellen/Views in sinnvoller Reihenfolge.
    """
    if not ad_ids:
        return pd.DataFrame(columns=["ad_id","perspective","confidence","rationale"])

    candidates = [
        "campaign_perspective_results",
        "ad_perspective_results",
        "ad_stance_results",
        "ad_perspective",        # falls alt
    ]
    for table in candidates:
        cols = _table_cols(conn, table)
        if not cols: 
            continue
        # Minimal: ad_id + perspective
        if "ad_id" not in cols or "perspective" not in cols:
            continue
        conf_col = "confidence" if "confidence" in cols else None
        rat_col  = "rationale" if "rationale" in cols else None
        sel_cols = ["ad_id","perspective"] + ([conf_col] if conf_col else []) + ([rat_col] if rat_col else [])
        sql = f"SELECT {', '.join(sel_cols)} FROM {table} WHERE ad_id = ANY(%s)"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (ad_ids,))
                rows = cur.fetchall()
                out_cols = [d[0] for d in cur.description]
            df = pd.DataFrame([dict(zip(out_cols, r)) for r in rows])
            if not df.empty:
                if "confidence" not in df.columns: df["confidence"] = np.nan
                if "rationale" not in df.columns: df["rationale"] = None
                # Normalisieren
                df["perspective"] = df["perspective"].astype(str).str.lower().map({
                    "pro":"pro","contra":"contra","neutral":"neutral","unklar":"unklar",
                    "unknown":"unklar","na":"unklar","n/a":"unklar"
                }).fillna("unklar")
                return df[["ad_id","perspective","confidence","rationale"]]
        except Exception:
            try: conn.rollback()
            except Exception: pass
            continue

    return pd.DataFrame(columns=["ad_id","perspective","confidence","rationale"])

def fetch_campaign_perspective_for_ad(conn, campaign_slug: str, page_name: str) -> Optional[dict]:
    """
    Holt (falls vorhanden) die Perspektiven-Klassifizierung aus campaign_perspective_results
    für die Kombination (campaign_slug, page_name).
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stance, confidence, rationale_bullets
                FROM campaign_perspective_results
                WHERE campaign_slug = %s AND page_name = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (campaign_slug, page_name),
            )
            row = cur.fetchone()
    except Exception:
        try: conn.rollback()
        except Exception: pass
        row = None

    if not row:
        return None

    stance, conf, bullets = row
    # Normalisieren wie in get_perspectives_for_ads
    stance = (str(stance).strip().lower()
              .replace("unknown", "unklar"))
    if stance not in {"pro","contra","neutral","unklar"}:
        stance = "unklar"

    return {
        "perspective": stance,
        "confidence": float(conf) if conf is not None else None,
        "rationale": "\n".join(bullets) if isinstance(bullets, (list, tuple)) else (bullets or None),
        "source": "campaign_perspective_results",
    }

# -------------------------------------------------------------------
# DB-Verbindung + Masterliste
# -------------------------------------------------------------------
try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")
    st.stop()

master = list_fully_analyzed_ads(conn)
if master.empty:
    st.info("Es gibt noch keine vollständig analysierten Ads (fused + Topics + Weaknesses).")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Einzel-Ad", "Akteur/Gruppe", "Kampagne (gesamt)"])

# ===================================================================
# TAB 1 – EINZEL-AD
# ===================================================================
with tab1:
    with st.expander("Filter", expanded=True):
        # 1) Kampagne
        camp_opts = (
            master[["campaign_slug","campaign_name"]]
            .drop_duplicates()
            .sort_values("campaign_name")
            .assign(label=lambda d: d["campaign_name"] + " (" + d["campaign_slug"] + ")")
        )
        sel_camps = st.multiselect(
            "Kampagne",
            options=camp_opts["campaign_slug"].tolist(),
            default=camp_opts["campaign_slug"].tolist(),
            format_func=lambda slug: camp_opts.set_index("campaign_slug").loc[slug, "label"],
            key="t1_camps",
        )
        df_c = master[master["campaign_slug"].isin(sel_camps)] if sel_camps else master.iloc[0:0]

        # 2) Akteur/Gruppe
        group_opts = sorted([g for g in df_c["page_name"].dropna().unique().tolist() if str(g).strip()])
        sel_groups = st.multiselect("Akteur/Gruppe (page_name)", options=group_opts, key="t1_groups", default=[])
        df_g = df_c[df_c["page_name"].isin(sel_groups)] if sel_groups else df_c

        # 3) Ad
        ad_opts = df_g.sort_values(["campaign_name","page_name","ad_id"], ascending=[True,True,False])
        if ad_opts.empty:
            st.warning("Keine Ads für diese Filter.")
            st.stop()
        label2id = dict(zip(ad_opts["label"], ad_opts["ad_id"]))
        sel_label = st.selectbox("Ad", options=ad_opts["label"].tolist(), key="t1_ad")
        sel_ad_id = int(label2id[sel_label])

    # --- Daten laden
    hist = fused_history_for_ad(conn, sel_ad_id)
    fused_latest = hist.tail(1).iloc[0]["fused"] if not hist.empty else {}
    api_raw = ((fused_latest or {}).get("api") or {}).get("raw") or {}
    llm_blocks = extract_llm_blocks(fused_latest)

    page_name = str(api_raw.get("page_name") or "—")
    bylines   = ", ".join(_norm_list(api_raw.get("bylines")))
    start     = api_raw.get("ad_delivery_start_time") or api_raw.get("ad_creation_time")
    stop      = api_raw.get("ad_delivery_stop_time") or "offen"
    platforms = ", ".join(api_raw.get("publisher_platforms") or [])

    weak = fetch_weakness_for_ad(conn, sel_ad_id, model=None)
    overall_risk = float(weak.get("overall_risk") or np.nan) if weak else np.nan
    overall_conf = float(weak.get("overall_confidence") or np.nan) if weak else np.nan
    weak_payload = (weak or {}).get("payload") or {}

    topics = fetch_topics_for_ad(conn, sel_ad_id) or {}
    vis = llm_blocks["vis"] or {}
    txt = llm_blocks["txt"] or {}
    sem = llm_blocks["sem"] or {}
    cta_typ = txt.get("cta_typ", "Unklar")

    # Screenshot
    left, right = st.columns([1, 1])
    mid, b64, mime, caption = fetch_screenshot_for_fused(conn, sel_ad_id, fused_latest)
    with left:
        if b64 and mime:
            st.image(f"data:{mime};base64,{b64}", caption=caption, use_container_width=True)
        else:
            st.info("Kein Screenshot verfügbar.")

    # Prompt + Button
    def build_ad_summary_prompt(
        *, ad_id:int, page_name:str, bylines:str, start:str, stop:str, platforms:str,
        topics:dict, vis:Dict[str,Any], txt:Dict[str,Any], sem:Dict[str,Any],
        overall_risk:Optional[float], overall_conf:Optional[float]
    ) -> str:
        bodies = api_raw.get("ad_creative_bodies") or api_raw.get("ad_creative_body") or []
        if isinstance(bodies, str): bodies = [bodies]
        risk_rows=[]
        for k,nice in {
            "factual_accuracy":"Faktischer Gehalt","framing_quality":"Framing",
            "visual_mislead":"Visuals","targeting_risks":"Targeting","policy_legal":"Policy/Recht",
            "transparency_context":"Transparenz","consistency_history":"Konsistenz",
            "usability_accessibility":"Usability/Barrierefreiheit",
        }.items():
            sc = weak_payload.get(f"score_{k}"); rat = weak_payload.get(f"rationale_{k}")
            if sc is not None: risk_rows.append((nice, float(sc), (rat or "")))
        risk_rows.sort(key=lambda t: t[1], reverse=True)
        risk_top = [{"kategorie":k,"score":s,"grund":(g[:220] if g else "")} for k,s,g in risk_rows[:2]]

        demo_df = extract_demography(api_raw).assign(label=lambda d: d["gender"].str.title()+" "+d["age"])
        regs_df = extract_regions(api_raw)
        def _top_series(df: pd.DataFrame, key: str) -> List[str]:
            if df is None or df.empty: return []
            tmp = (df.groupby(key, as_index=False)["spend"].sum().sort_values("spend", ascending=False).head(3))
            return [f"{r[key]} (~{r['spend']:.0f})" for _,r in tmp.iterrows()]
        demo_top = _top_series(demo_df, "label")
        regs_top = _top_series(regs_df, "region")

        prompt = f"""
Du bist Analyst:in für politische Online-Werbung in der Schweiz. Fasse die folgende **einzelne Ad** prägnant in **3–5 Sätzen** zusammen – neutral, faktenbasiert.

Kontext
- Ad-ID: {ad_id}
- Gruppe/Akteur: {page_name}
- Sponsoren/Bylines: {bylines or "—"}
- Laufzeit: {str(start)[:10]} → {str(stop)[:10]}
- Plattform(en): {platforms or "—"}

Inhalte & Stil
- Ad-Text (gekürzt): {" ".join(bodies)[:600] if bodies else "—"}
- Themen (Tagging): {", ".join(topics.get('topics', [])) if topics else "—"}
- CTA: Typ={txt.get("cta_typ","Unklar")}, Prominenz={txt.get("cta_visuelle_prominenz","Unklar")}, Position={txt.get("cta_position","—")}
- Layout/Komposition: {vis.get("dominante_layoutstruktur","?")} / {vis.get("kompositionstyp","?")}; Plattform (LLM)={vis.get("plattform","Unbekannt")}
- Farbpalette (max 5): {", ".join([c for c in (vis.get("farbpalette") or [])][:5]) or "—"}
- Semantik (Kurz): Zielgruppe={sem.get("zielgruppe","—")}; Framing={sem.get("framing_typ","—")}; Appell={sem.get("emotionaler_apell","—")}; Versprechen={sem.get("werbeversprechen","—")}

Targeting
- Demografie Top: {", ".join(demo_top) or "—"}
- Regionen Top: {", ".join(regs_top) or "—"}

Schwachstellen & Risiken
- Gesamtrisiko: {overall_risk if overall_risk is not None else "—"} · Gesamt-Konfidenz: {overall_conf if overall_conf is not None else "—"}
- Top-Risiken: {json.dumps(risk_top, ensure_ascii=False)}

Aufgabe
- Schreibe 3–5 Sätze zu Inhalt/Standpunkt, Stil/Creative, auffälligen Zielgruppen/Regionen und den wichtigsten Risiken.
""".strip()
        return prompt

    prompt = build_ad_summary_prompt(
        ad_id=sel_ad_id, page_name=page_name, bylines=bylines,
        start=start, stop=stop, platforms=platforms, topics=topics,
        vis=vis, txt=txt, sem=sem,
        overall_risk=None if math.isnan(overall_risk) else overall_risk,
        overall_conf=None if math.isnan(overall_conf) else overall_conf,
    )

    with right:
        st.markdown("#### LLM-Kurzbeschreibung (3–5 Sätze)")
        if st.button("🔁 Kurzbeschreibung erzeugen", key="t1_sum_btn", use_container_width=True):
            summary_text = None
            if _HAVE_OPENAI and os.getenv("OPENAI_API_KEY","").strip():
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini", temperature=0.2,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    summary_text = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    st.warning(f"Automatische Zusammenfassung nicht möglich ({e}).")
            if not summary_text:
                summary_text = f"Die Ad stammt von {page_name}. CTA: {cta_typ or 'Unklar'}. " + \
                               (f"Gesamtrisiko {overall_risk:.2f}." if not math.isnan(overall_risk) else "")
            st.write(summary_text)
        with st.expander("Prompt (zur Nachvollziehbarkeit)"):
            st.code(prompt, language="markdown")

    # Header KPIs
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Gesamtrisiko", f"{overall_risk:.2f}" if not math.isnan(overall_risk) else "–")
    with c2: st.metric("Konfidenz", f"{overall_conf:.2f}" if not math.isnan(overall_conf) else "–")
    with c3: st.metric("CTA-Typ", cta_typ)
    st.caption(f"**Gruppe:** {page_name} · **Sponsoren:** {bylines or '—'} · **Zeitraum:** {str(start)[:10]} → {str(stop)[:10]} · **Plattformen:** {platforms or '—'}")

    # Perspektive (Klassifizierung) – EINZEL
    st.subheader("Perspektive (Klassifizierung)")

    ad_stance = get_perspectives_for_ads(conn, [sel_ad_id])

    if not ad_stance.empty:
        row = ad_stance.iloc[0]
        colA, colB = st.columns(2)
        with colA:
            st.metric("Perspektive", str(row["perspective"]).title())
        with colB:
            try:
                st.metric("Konfidenz", f"{float(row['confidence']):.2f}")
            except Exception:
                st.metric("Konfidenz", "–")
        if isinstance(row.get("rationale"), str) and row["rationale"].strip():
            with st.expander("Begründung (Modell)"):
                st.write(row["rationale"])
    else:
        # ---- Fallback: campaign_perspective_results über (campaign_slug, page_name)
        try:
            camp_slug = str(
                master.loc[master["ad_id"] == sel_ad_id, "campaign_slug"].iloc[0]
            )
        except Exception:
            camp_slug = None

        if camp_slug:
            fallback = fetch_campaign_perspective_for_ad(conn, camp_slug, str(page_name))
        else:
            fallback = None

        if fallback:
            colA, colB = st.columns(2)
            with colA:
                st.metric("Perspektive (Kampagne/Gruppe)", fallback["perspective"].title())
            with colB:
                st.metric(
                    "Konfidenz",
                    f"{float(fallback['confidence']):.2f}"
                    if fallback["confidence"] is not None else "–",
                )
            if isinstance(fallback.get("rationale"), str) and fallback["rationale"].strip():
                with st.expander("Begründung (Kampagnen-Modell)"):
                    st.write(fallback["rationale"])
            st.caption("Quelle: campaign_perspective_results (gemappt auf Kampagne & Gruppe).")
        else:
            st.caption("Keine Perspektivenklassifizierung gefunden.")


    # Themen (Tagging) – EINZEL
    st.subheader("Themen (Tagging)")
    if topics:
        toks = topics.get("topics", [])
        st.write(", ".join(map(str, toks)) if toks else "—")
        if topics.get("rationale_bullets"):
            with st.expander("Begründungen (Bullets)"):
                st.markdown("\n".join(f"- {b}" for b in topics["rationale_bullets"]))
        st.caption(f"Konfidenz: {topics.get('confidence','—')}  ·  Modell: {topics.get('model','—')}")
    else:
        st.caption("Keine Tagging-Ergebnisse gefunden.")

    # Risiken
    if weak_payload:
        st.subheader("Risiko-Kategorien")
        cats = [
            "factual_accuracy","framing_quality","visual_mislead","targeting_risks",
            "policy_legal","transparency_context","consistency_history","usability_accessibility",
        ]
        rows = [{"Kategorie": k, "Score": float(weak_payload.get(f"score_{k}"))}
                for k in cats if weak_payload.get(f"score_{k}") is not None]
        if rows:
            st.plotly_chart(px.bar(pd.DataFrame(rows), x="Kategorie", y="Score"), use_container_width=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

    # Audience & Regionen
    st.subheader("Audience & Regionen")
    demo_df = extract_demography(api_raw)
    if not demo_df.empty:
        agg_aud = (demo_df.assign(label=lambda d: d["gender"].str.title()+" "+d["age"])
                   .groupby("label", as_index=False)["spend"].sum()
                   .sort_values("spend", ascending=False))
        st.plotly_chart(px.bar(agg_aud, x="label", y="spend", title="Budget nach Zielgruppe (mid)"),
                        use_container_width=True)
    else:
        st.caption("Keine demografische Verteilung verfügbar.")
    regs_df = extract_regions(api_raw)
    if not regs_df.empty:
        top = regs_df.sort_values("spend", ascending=False).head(15)
        st.plotly_chart(px.bar(top, x="region", y="spend", title="Top-Regionen (Spend)"),
                        use_container_width=True)
    else:
        st.caption("Keine regionale Verteilung verfügbar.")

    # Creative-Insights
    st.subheader("Creative-Insights")
    fl = vis.get("flächenverteilung") or {}
    if fl:
        sdf = pd.DataFrame([{
            "Text": _num(fl.get("textfläche")), "Bild": _num(fl.get("bildfläche")), "Weißraum": _num(fl.get("weißraum")),
        }]).melt(var_name="Typ", value_name="Anteil")
        st.plotly_chart(px.bar(sdf, x="Typ", y="Anteil", title="Flächenverteilung"), use_container_width=True)

    colA, colB, colC = st.columns(3)
    with colA: st.metric("Komposition", vis.get("kompositionstyp","?"))
    with colB: st.metric("Layout", vis.get("dominante_layoutstruktur","?"))
    with colC: st.metric("Plattform (LLM)", vis.get("plattform","Unbekannt"))

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("CTA-Typ", txt.get("cta_typ","Unklar"))
    with col2: st.metric("CTA-Prominenz", txt.get("cta_visuelle_prominenz","Unklar"))
    with col3: st.metric("CTA-Position", txt.get("cta_position","—"))

    pal = [c for c in (_norm_hex(str(x)) for x in (vis.get("farbpalette") or [])) if c]
    if pal:
        st.caption("Farbpalette")
        cols = st.columns(len(pal))
        for c, col in zip(pal, cols):
            col.markdown(f"<div style='height:24px;border-radius:4px;background:{c}'></div>", unsafe_allow_html=True)

    if sem:
        with st.expander("Semantik – Kurzüberblick"):
            bullets = [
                f"- **Werbeversprechen:** {sem.get('werbeversprechen','–')}",
                f"- **Zielgruppe (Modell):** {sem.get('zielgruppe','–')}",
                f"- **Ansprache:** {sem.get('ansprache_typ','–')}",
                f"- **Framing:** {sem.get('framing_typ','–')}",
                f"- **Emotionaler Appell:** {sem.get('emotionaler_apell','–')}",
            ]
            st.markdown("\n".join(bullets))

# ===================================================================
# TAB 2 – AKTEUR/GRUPPE (aggregate)
# ===================================================================
with tab2:
    # ------------- Filter -------------
    with st.expander("Filter", expanded=True):
        camp_opts = (
            master[["campaign_slug", "campaign_name"]]
            .drop_duplicates()
            .sort_values("campaign_name")
            .assign(label=lambda d: d["campaign_name"] + " (" + d["campaign_slug"] + ")")
        )
        sel_camps = st.multiselect(
            "Kampagne",
            options=camp_opts["campaign_slug"].tolist(),
            default=camp_opts["campaign_slug"].tolist(),
            format_func=lambda slug: camp_opts.set_index("campaign_slug").loc[slug, "label"],
            key="t2_camps",
        )

        df_c = master[master["campaign_slug"].isin(sel_camps)] if sel_camps else master.iloc[0:0]

        group_opts = sorted([g for g in df_c["page_name"].dropna().unique().tolist() if str(g).strip()])
        if not group_opts:
            st.warning("Keine Gruppen für diese Kampagnen.")
            st.stop()
        sel_group = st.selectbox("Akteur/Gruppe", options=group_opts, key="t2_group")

        # Selektion Gruppe → Ads
        df_g = df_c[df_c["page_name"] == sel_group]
        ad_ids = df_g["ad_id"].astype(int).tolist()
        if not ad_ids:
            st.info("Keine Ads für diese Gruppe.")
            st.stop()

    st.markdown(f"### Gruppe/Akteur: **{sel_group}**")

    # Anzahl analysierter Ads (innerhalb der aktuell gewählten Kampagnen)
    n_ads = df_g["ad_id"].nunique()
    n_ext = df_g["ad_external_id"].dropna().nunique()
    st.caption(f"Analysierte Ads (Gruppe, Auswahl oben): **{n_ads}**  ·  Distinct External-IDs: **{n_ext}**")

    # ------------- Screenshots (links/rechts) -------------
    shot_df = pd.DataFrame()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id AS media_id, m.ad_id, m.created_at, mb.mime_type, mb.b64
                FROM media m
                LEFT JOIN media_base64 mb ON mb.media_id = m.id
                WHERE m.kind='screenshot' AND m.ad_id = ANY(%s)
                ORDER BY m.created_at DESC
                """,
                (ad_ids,),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            shot_df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        shot_df = pd.DataFrame()

    # Label-Mapping wie Tab 1 – aber OHNE Zeitstempel: "Gruppenname (external_ad_id)"
    meta_map = df_g.set_index("ad_id")[["page_name", "ad_external_id"]].to_dict("index")

    if not shot_df.empty:
        shot_df = shot_df.reset_index(drop=True)
        shot_df["option_id"] = shot_df.index.astype(int)

        def _shot_label_without_ts(s: pd.Series) -> str:
            ad_id_i = int(s["ad_id"])
            meta = meta_map.get(ad_id_i, {})
            page = (meta.get("page_name") or f"Ad {ad_id_i}").strip()
            ext = str(meta.get("ad_external_id") or ad_id_i)
            return f"{page} ({ext})"

        label_map = dict(zip(shot_df["option_id"], shot_df.apply(_shot_label_without_ts, axis=1)))

        c1, c2 = st.columns(2)
        with c1:
            sel_id_left = st.selectbox(
                "Screenshots (links)",
                options=shot_df["option_id"].tolist(),
                format_func=lambda oid: label_map.get(int(oid), str(oid)),
                key="t2_shot1",
            )
            row1 = shot_df.set_index("option_id").loc[int(sel_id_left)]
            if row1.get("b64") and row1.get("mime_type"):
                st.image(f"data:{row1['mime_type']};base64,{row1['b64']}", use_container_width=True)

        with c2:
            sel_id_right = st.selectbox(
                "Screenshots (rechts)",
                options=shot_df["option_id"].tolist(),
                format_func=lambda oid: label_map.get(int(oid), str(oid)),
                key="t2_shot2",
            )
            row2 = shot_df.set_index("option_id").loc[int(sel_id_right)]
            if row2.get("b64") and row2.get("mime_type"):
                st.image(f"data:{row2['mime_type']};base64,{row2['b64']}", use_container_width=True)
    else:
        st.caption("Für diese Gruppe sind noch keine Screenshots gespeichert.")

    # ------------- Daten sammeln (fused + fallbacks) -------------
    fused_latest_by_ad: dict[int, dict] = {}
    api_raw_by_ad: dict[int, dict] = {}
    for ad_id in ad_ids:
        hist = fused_history_for_ad(conn, ad_id)
        if not hist.empty:
            fused_latest = hist.tail(1).iloc[0]["fused"]
            fused_latest_by_ad[ad_id] = fused_latest
            api_raw_by_ad[ad_id] = ((fused_latest or {}).get("api") or {}).get("raw") or {}
        else:
            fused_latest_by_ad[ad_id] = {}
            api_raw_by_ad[ad_id] = {}

    # ------------- LLM-Kurzbeschreibung (vor Perspektiven) -------------
    st.markdown("#### LLM-Kurzbeschreibung (Gruppe, 3–5 Sätze)")

    # Themen/CTA für Prompt
    all_topics: list[str] = []
    for ad_id in ad_ids:
        t = fetch_topics_for_ad(conn, ad_id) or {}
        all_topics.extend([str(x) for x in (t.get("topics") or [])])

    top_topics = []
    if all_topics:
        tdf = pd.Series(all_topics).value_counts().head(5)
        top_topics = [f"{idx} (n={val})" for idx, val in tdf.items()]

    cta_vals = []
    for ad_id in ad_ids:
        llm = extract_llm_blocks(fused_latest_by_ad.get(ad_id, {})) if fused_latest_by_ad.get(ad_id) else {}
        txt = (llm or {}).get("txt") or {}
        cta_vals.append(txt.get("cta_typ") or "Unklar")
    cta_counts = pd.Series(cta_vals).value_counts().to_dict() if cta_vals else {}

    group_prompt = f"""
Du bist Analyst:in für politische Online-Werbung in der Schweiz. Fasse die **Gruppe/Akteur '{sel_group}'** über alle vorliegenden Ads prägnant in **3–5 Sätzen** zusammen – neutral, faktenbasiert.

Kontext
- Anzahl analysierter Ads: {len(ad_ids)}
- Top-Themen: {", ".join(top_topics) if top_topics else "—"}
- CTA-Verteilung: {", ".join([f"{k}: {v}" for k, v in cta_counts.items()]) if cta_counts else "—"}
""".strip()

    if st.button("🔁 Gruppen-Zusammenfassung erzeugen", key="t2_sum_btn"):
        grp_summary = None
        if _HAVE_OPENAI and os.getenv("OPENAI_API_KEY", "").strip():
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[{"role": "user", "content": group_prompt}],
                )
                grp_summary = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                st.warning(f"Automatische Gruppen-Zusammenfassung nicht möglich ({e}).")
        st.write(grp_summary or f"{sel_group}: {len(ad_ids)} Ads; Themen: {', '.join(top_topics) if top_topics else '—'}.")

    with st.expander("Prompt (zur Nachvollziehbarkeit)"):
        st.code(group_prompt, language="markdown")

    # ------------- Perspektiven (Gruppe) -------------
    st.subheader("Perspektiven (Gruppe)")

    # (A) Kampagnenweite Perspektive (campaign_perspective_results) für diese Gruppe
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stance, confidence, topic, rationale_bullets
                FROM campaign_perspective_results
                WHERE campaign_slug = ANY(%s) AND page_name = %s
                """,
                (sel_camps, sel_group),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if rows else ["stance", "confidence", "topic", "rationale_bullets"]
        camp_persp = pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(columns=cols)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        camp_persp = pd.DataFrame(columns=["stance", "confidence", "topic", "rationale_bullets"])

    if not camp_persp.empty:
        dist = camp_persp.groupby("stance", as_index=False).size().rename(columns={"size": "anzahl"})
        st.plotly_chart(px.bar(dist, x="stance", y="anzahl", title="Verteilung (campaign_perspective_results)"),
                        use_container_width=True)
        with st.expander("Einzel-Einträge (Kampagnen-Perspektiven)"):
            st.dataframe(
                camp_persp.assign(confidence=lambda d: d["confidence"].astype(float)).sort_values("confidence", ascending=False),
                use_container_width=True, height=260
            )
    else:
        st.caption("Keine Einträge in campaign_perspective_results für diese Gruppe.")

    # (B) Einzel-Ad Perspektiven
    stance_df = get_perspectives_for_ads(conn, ad_ids)
    if not stance_df.empty:
        dist2 = stance_df.groupby("perspective", as_index=False).size().rename(columns={"size": "anzahl"})
        st.plotly_chart(px.bar(dist2, x="perspective", y="anzahl", title="Verteilung (Einzel-Ads)"),
                        use_container_width=True)
        with st.expander("Einzel-Ad Perspektiven"):
            st.dataframe(
                stance_df.sort_values("confidence", ascending=False, na_position="last"),
                use_container_width=True, height=260
            )
    else:
        st.caption("Keine Perspektivenklassifizierung für die Einzel-Ads gefunden.")

    # ------------- Themen (Tagging) -------------
    st.subheader("Themen (Gruppe)")
    topic_rows = []
    for ad_id in ad_ids:
        t = fetch_topics_for_ad(conn, ad_id) or {}
        for x in (t.get("topics") or []):
            topic_rows.append({"ad_id": ad_id, "topic": str(x)})

    if topic_rows:
        tdf = pd.DataFrame(topic_rows)
        top = tdf["topic"].value_counts().reset_index()
        top.columns = ["topic", "anzahl"]
        st.plotly_chart(px.bar(top.head(20), x="topic", y="anzahl", title="Top-Themen (Anzahl Ads)"),
                        use_container_width=True)
        with st.expander("Themen nach Ad"):
            by_ad = (
                tdf.groupby("ad_id")["topic"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()
            )
            st.dataframe(by_ad, use_container_width=True, height=260)
    else:
        st.caption("Keine Themenklassifizierung vorhanden.")

    # ------------- Risiken/Schwachstellen (aggregiert) -------------
    st.subheader("Risiko-Kategorien (Ø Gruppe)")
    risk_cats = [
        "factual_accuracy", "framing_quality", "visual_mislead", "targeting_risks",
        "policy_legal", "transparency_context", "consistency_history", "usability_accessibility",
    ]
    risk_rows = []
    for ad_id in ad_ids:
        w = fetch_weakness_for_ad(conn, ad_id)
        if not w:
            continue
        payload = w.get("payload") or {}
        for k in risk_cats:
            sc = payload.get(f"score_{k}")
            if sc is not None:
                risk_rows.append({"Kategorie": k, "Score": float(sc)})

    if risk_rows:
        rdf = (
            pd.DataFrame(risk_rows)
            .groupby("Kategorie", as_index=False)["Score"].mean()
            .sort_values("Score", ascending=False)
        )
        st.plotly_chart(px.bar(rdf, x="Kategorie", y="Score", title="Risiko-Kategorien (Ø)"),
                        use_container_width=True)
        st.dataframe(rdf, use_container_width=True, height=240)
    else:
        st.caption("Keine Schwachstellen-Ergebnisse gefunden.")

    # ------------- Audience & Regionen (aggregiert) -------------
    st.subheader("Audience & Regionen (aggregiert)")
    demo_parts, regs_parts = [], []
    for ad_id in ad_ids:
        api_raw = api_raw_by_ad.get(ad_id, {})
        ddf = extract_demography(api_raw)
        if not ddf.empty:
            ddf = ddf.assign(label=lambda d: d["gender"].str.title() + " " + d["age"])
            demo_parts.append(ddf[["label", "spend"]])
        rdf = extract_regions(api_raw)
        if not rdf.empty:
            regs_parts.append(rdf[["region", "spend"]])

    if demo_parts:
        agg_demo = (
            pd.concat(demo_parts, ignore_index=True)
            .groupby("label", as_index=False)["spend"].sum()
            .sort_values("spend", ascending=False)
            .head(20)
        )
        st.plotly_chart(px.bar(agg_demo, x="label", y="spend", title="Budget nach Zielgruppe (Summe)"),
                        use_container_width=True)
    else:
        st.caption("Keine demografische Verteilung verfügbar.")

    if regs_parts:
        agg_regs = (
            pd.concat(regs_parts, ignore_index=True)
            .groupby("region", as_index=False)["spend"].sum()
            .sort_values("spend", ascending=False)
            .head(20)
        )
        st.plotly_chart(px.bar(agg_regs, x="region", y="spend", title="Top-Regionen (Summe)"),
                        use_container_width=True)
    else:
        st.caption("Keine regionale Verteilung verfügbar.")

    # ------------- Creative-Insights (aggregiert) -------------
    st.subheader("Creative-Insights (Gruppe)")
    # CTA-Verteilung
    if cta_vals:
        cta_df = pd.Series(cta_vals, name="CTA").value_counts().reset_index()
        cta_df.columns = ["CTA", "anzahl"]
        st.plotly_chart(px.bar(cta_df, x="CTA", y="anzahl", title="CTA-Verteilung"),
                        use_container_width=True)

    # Flächenverteilung (Ø) + weitere visuelle Merkmale
    fl_rows = []
    comp_vals, layout_vals, platform_vals = [], [], []
    palette_vals = []
    for ad_id in ad_ids:
        llm_blocks = extract_llm_blocks(fused_latest_by_ad.get(ad_id, {})) if fused_latest_by_ad.get(ad_id) else {}
        vis = (llm_blocks or {}).get("vis") or {}

        fl = vis.get("flächenverteilung") or {}
        if fl:
            fl_rows.append({
                "Text": _num(fl.get("textfläche")),
                "Bild": _num(fl.get("bildfläche")),
                "Weißraum": _num(fl.get("weißraum")),
            })

        if vis.get("kompositionstyp"):
            comp_vals.append(vis.get("kompositionstyp"))
        if vis.get("dominante_layoutstruktur"):
            layout_vals.append(vis.get("dominante_layoutstruktur"))
        if vis.get("plattform"):
            platform_vals.append(vis.get("plattform"))
        for c in (vis.get("farbpalette") or []):
            c2 = _norm_hex(str(c))
            if c2:
                palette_vals.append(c2)

    if fl_rows:
        fl_df = pd.DataFrame(fl_rows)
        avg = fl_df.mean(numeric_only=True).reset_index()
        avg.columns = ["Typ", "Anteil"]
        st.plotly_chart(px.bar(avg, x="Typ", y="Anteil", title="Flächenverteilung (Ø)"),
                        use_container_width=True)

    colsA, colsB, colsC = st.columns(3)
    if comp_vals:
        comp_df = pd.Series(comp_vals, name="Komposition").value_counts().reset_index()
        comp_df.columns = ["Komposition", "anzahl"]
        with colsA:
            st.plotly_chart(px.bar(comp_df, x="Komposition", y="anzahl", title="Komposition"),
                            use_container_width=True)
    if layout_vals:
        lay_df = pd.Series(layout_vals, name="Layout").value_counts().reset_index()
        lay_df.columns = ["Layout", "anzahl"]
        with colsB:
            st.plotly_chart(px.bar(lay_df, x="Layout", y="anzahl", title="Layout"),
                            use_container_width=True)
    if platform_vals:
        plat_df = pd.Series(platform_vals, name="Plattform").value_counts().reset_index()
        plat_df.columns = ["Plattform", "anzahl"]
        with colsC:
            st.plotly_chart(px.bar(plat_df, x="Plattform", y="anzahl", title="Plattform (LLM)"),
                            use_container_width=True)

    if palette_vals:
        st.caption("Top-Farbpalette (Häufigkeit)")
        pal_df = pd.Series(palette_vals, name="farbe").value_counts().head(10).reset_index()
        pal_df.columns = ["farbe", "anzahl"]
        cols = st.columns(len(pal_df))
        for (_, r), col in zip(pal_df.iterrows(), cols):
            col.markdown(f"<div style='height:24px;border-radius:4px;background:{r['farbe']}'></div>", unsafe_allow_html=True)
            col.caption(f"{r['farbe']} ({int(r['anzahl'])})")

# ---------------- Zeitverlauf (Gruppe) — API-Snapshots, Summe über alle Ads der Gruppe ----------------
st.subheader("Zeitverlauf (Gruppe)")

# 1) External-IDs + Metadaten der gruppen-Ads (nur die, die in ad_llm_fused/master vorkommen = analysiert)
df_grp_ads = (
    df_g[["ad_id", "ad_external_id", "page_name", "campaign_id"]]
    .dropna(subset=["ad_id", "ad_external_id"])
    .astype({"ad_id": int})
)
# sauberer Set-Aufbau (leere Strings raus)
ext_ids = set(df_grp_ads["ad_external_id"].dropna().astype(str).str.strip())
ext_ids.discard("")
if not ext_ids:
    st.caption("Keine External-IDs für die Gruppe gefunden.")
else:
    # 2) Alle relevanten api_snapshots der betroffenen Kampagnen laden
    try:
        campaign_ids = df_grp_ads["campaign_id"].dropna().astype(int).unique().tolist()
    except Exception:
        campaign_ids = []

    api_df = pd.DataFrame(columns=["snapshot_date", "raw"])
    if campaign_ids:
        try:
            with conn.cursor() as cur:
                cols = _table_cols(conn, "api_snapshots")
                json_candidates = ["payload", "raw", "api_raw", "snapshot"]
                ts_candidates   = ["created_at", "ingested_at", "fetched_at", "updated_at"]
                date_col = "snapshot_date" if "snapshot_date" in cols else None
                json_col = next((c for c in json_candidates if c in cols), None)
                ts_col   = next((c for c in ts_candidates   if c in cols), None)
                if not json_col:
                    st.caption("In api_snapshots wurde keine JSON-Spalte (payload/raw/…) gefunden.")
                else:
                    if date_col:
                        sql = f"""
                            SELECT {date_col} AS snapshot_date, {json_col} AS raw
                            FROM api_snapshots
                            WHERE campaign_id = ANY(%s)
                            ORDER BY {date_col}
                        """
                    else:
                        if not ts_col:
                            ts_col = "created_at"
                        sql = f"""
                            SELECT DATE({ts_col}) AS snapshot_date, {json_col} AS raw
                            FROM api_snapshots
                            WHERE campaign_id = ANY(%s)
                            ORDER BY DATE({ts_col})
                        """
                    cur.execute(sql, (campaign_ids,))
                    rows = cur.fetchall()
                    if rows:
                        cols_out = [d[0] for d in cur.description]
                        api_df = pd.DataFrame([dict(zip(cols_out, r)) for r in rows])
        except Exception:
            try: conn.rollback()
            except Exception: pass
            api_df = pd.DataFrame(columns=["snapshot_date", "raw"])

    # 3) Payload je Snapshot durchsuchen, passende Ads summieren
    import numpy as np
    import plotly.graph_objects as go

    ts_agg = {}  # date -> {"spend":..., "impressions":..., "eu_total_reach":..., "ads": set()}

    if not api_df.empty:
        def _parse_json(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except Exception: return []
            return x or []
        api_df["raw"] = api_df["raw"].apply(_parse_json)
        api_df["snapshot_date"] = pd.to_datetime(api_df["snapshot_date"], errors="coerce").dt.date
        api_df = api_df[api_df["snapshot_date"] != _date(2025, 9, 10)]

        for r in api_df.itertuples():
            ads_list = getattr(r, "raw", []) or []
            if not isinstance(ads_list, (list, tuple)):
                continue

            # Tagesaggregate nur anlegen, wenn mindestens 1 Gruppen-Ad im Snapshot ist
            day_spend = 0.0
            day_impr  = 0.0
            day_eu    = 0.0
            day_ads   = set()

            for item in ads_list:
                try:
                    ad_ext = str((item or {}).get("id") or "").strip()
                except Exception:
                    ad_ext = ""
                if not ad_ext or ad_ext not in ext_ids:
                    continue

                day_spend += float(_mid_value((item or {}).get("spend") or {}) or 0.0)
                day_impr  += float(_mid_value((item or {}).get("impressions") or {}) or 0.0)
                day_eu    += float((item or {}).get("eu_total_reach") or 0)
                day_ads.add(ad_ext)

            if day_ads:  # <-- nur Tage mit Gruppen-Ads übernehmen
                ts_agg[r.snapshot_date] = {
                    "spend": day_spend,
                    "impressions": day_impr,
                    "eu_total_reach": day_eu,
                    "ads": day_ads,
                }
    # DataFrame bauen
    if ts_agg:
        rows = []
        for d, vals in ts_agg.items():
            rows.append({
                "date": d,
                "spend": vals["spend"],
                "impressions": vals["impressions"],
                "eu_total_reach": vals["eu_total_reach"],
                "n_ads": len(vals["ads"]),
                "ads_list": sorted(list(vals["ads"])),
            })
        ts_sum = pd.DataFrame(rows).sort_values("date")
    else:
        ts_sum = pd.DataFrame(columns=["date", "spend", "impressions", "eu_total_reach", "n_ads", "ads_list"])

    if ts_sum.empty:
        st.caption("Keine spend/impressions/eu_total_reach in api_snapshots für diese Gruppe gefunden.")
    else:
        # Anzeige-Strings für Hover (gekürzt) – volles Listing steht unterhalb
        def _short_list(lst, n=6):
            if not lst: return "—"
            if len(lst) <= n: return ", ".join(lst)
            return ", ".join(lst[:n]) + f" … (+{len(lst)-n})"
        ts_sum["ads_short"] = ts_sum["ads_list"].apply(_short_list)

        # Spend
        fig_spend = go.Figure()
        fig_spend.add_trace(go.Scatter(
            x=ts_sum["date"], y=ts_sum["spend"], mode="lines+markers", name="Spend",
            customdata=np.stack([ts_sum["n_ads"], ts_sum["ads_short"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Spend (Summe): %{y:.0f} CHF"
                          "<br>Beiträge von %{customdata[0]} Ads"
                          "<br><span style='font-size:11px'>%{customdata[1]}</span>"
                          "<extra></extra>"
        ))
        fig_spend.update_layout(
            title="Spend über Zeit (API-Snapshots; Summe Gruppe)",
            xaxis_title="Datum", yaxis_title="Spend (CHF)",
            showlegend=False
        )
        st.plotly_chart(fig_spend, use_container_width=True)

        # Impressions
        fig_impr = go.Figure()
        fig_impr.add_trace(go.Scatter(
            x=ts_sum["date"], y=ts_sum["impressions"], mode="lines+markers", name="Impressions",
            customdata=np.stack([ts_sum["n_ads"], ts_sum["ads_short"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Impressions (Summe): %{y:.0f}"
                          "<br>Beiträge von %{customdata[0]} Ads"
                          "<br><span style='font-size:11px'>%{customdata[1]}</span>"
                          "<extra></extra>"
        ))
        fig_impr.update_layout(
            title="Impressions über Zeit (API-Snapshots; Summe Gruppe)",
            xaxis_title="Datum", yaxis_title="Impressions",
            showlegend=False
        )
        st.plotly_chart(fig_impr, use_container_width=True)

        # EU-Reach
        fig_eu = go.Figure()
        fig_eu.add_trace(go.Scatter(
            x=ts_sum["date"], y=ts_sum["eu_total_reach"], mode="lines+markers", name="EU-Reach",
            customdata=np.stack([ts_sum["n_ads"], ts_sum["ads_short"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>EU-Reach (Summe): %{y:.0f}"
                          "<br>Beiträge von %{customdata[0]} Ads"
                          "<br><span style='font-size:11px'>%{customdata[1]}</span>"
                          "<extra></extra>"
        ))
        fig_eu.update_layout(
            title="EU-Reach über Zeit (API-Snapshots; Summe Gruppe)",
            xaxis_title="Datum", yaxis_title="EU-Reach",
            showlegend=False
        )
        st.plotly_chart(fig_eu, use_container_width=True)

        # Vollständige Liste der verwendeten Ads (Validierung)
        st.markdown("**Verwendete Ads (in den Snapshots gefunden):**")
        used_all_ext = sorted(set(e for lst in ts_sum["ads_list"] for e in lst))
        if used_all_ext:
            meta_idx = df_grp_ads.set_index("ad_external_id")
            bullets = []
            for ext in used_all_ext:
                if ext in meta_idx.index:
                    row = meta_idx.loc[ext]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    label = f"{row['page_name']} ({ext}) — Ad {int(row['ad_id'])}"
                else:
                    label = f"{sel_group} ({ext})"
                bullets.append(f"- {label}")
            st.markdown("\n".join(bullets))
        else:
            st.caption("Keine passenden Ads in den Snapshots gefunden.")

# ===================================================================
# TAB 3 – KAMPAGNE (gesamt)  —  wie Tab 2, aber über ALLE Gruppen der Kampagne
# ===================================================================
with tab3:
    # ------------- Filter -------------
    with st.expander("Filter", expanded=True):
        camp_opts = (
            master[["campaign_slug","campaign_name","campaign_id"]]
            .drop_duplicates()
            .sort_values("campaign_name")
            .assign(label=lambda d: d["campaign_name"] + " (" + d["campaign_slug"] + ")")
        )
        if camp_opts.empty:
            st.stop()
        sel_camp = st.selectbox(
            "Kampagne",
            options=camp_opts["campaign_slug"].tolist(),
            format_func=lambda slug: camp_opts.set_index("campaign_slug").loc[slug, "label"],
            key="t3_camp",
        )

        # alle Ads der Kampagne (analysierte, aus master)
        df_k = master[master["campaign_slug"] == sel_camp].copy()
        ad_ids = df_k["ad_id"].astype(int).tolist()
        if not ad_ids:
            st.info("Keine analysierten Ads für diese Kampagne.")
            st.stop()

    camp_name  = camp_opts.set_index("campaign_slug").loc[sel_camp, "campaign_name"]
    st.markdown(f"### Kampagne: **{camp_name}**")

    # Kennzahlen (analog zu Tab 2)
    n_ads  = df_k["ad_id"].nunique()
    n_ext  = df_k["ad_external_id"].dropna().nunique()
    n_grp  = df_k["page_name"].dropna().nunique()
    st.caption(f"Analysierte Ads: **{n_ads}**  ·  Distinct External-IDs: **{n_ext}**  ·  Gruppen/Akteure: **{n_grp}**")

    # ------------- Screenshots (links/rechts) -------------
    shot_df = pd.DataFrame()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id AS media_id, m.ad_id, m.created_at, mb.mime_type, mb.b64
                FROM media m
                LEFT JOIN media_base64 mb ON mb.media_id = m.id
                WHERE m.kind='screenshot' AND m.ad_id = ANY(%s)
                ORDER BY m.created_at DESC
                """,
                (ad_ids,),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            shot_df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    except Exception:
        try: conn.rollback()
        except Exception: pass
        shot_df = pd.DataFrame()

    # Label-Mapping wie Tab 2 – OHNE Zeitstempel: "PageName (external_ad_id)"
    meta_map = df_k.set_index("ad_id")[["page_name", "ad_external_id"]].to_dict("index")

    if not shot_df.empty:
        shot_df = shot_df.reset_index(drop=True)
        shot_df["option_id"] = shot_df.index.astype(int)

        def _shot_label_without_ts(s: pd.Series) -> str:
            ad_id_i = int(s["ad_id"])
            meta = meta_map.get(ad_id_i, {})
            page = (meta.get("page_name") or f"Ad {ad_id_i}").strip()
            ext  = str(meta.get("ad_external_id") or ad_id_i)
            return f"{page} ({ext})"

        label_map = dict(zip(shot_df["option_id"], shot_df.apply(_shot_label_without_ts, axis=1)))

        c1, c2 = st.columns(2)
        with c1:
            sel_id_left = st.selectbox(
                "Screenshots (links)",
                options=shot_df["option_id"].tolist(),
                format_func=lambda oid: label_map.get(int(oid), str(oid)),
                key="t3_shot1",
            )
            row1 = shot_df.set_index("option_id").loc[int(sel_id_left)]
            if row1.get("b64") and row1.get("mime_type"):
                st.image(f"data:{row1['mime_type']};base64,{row1['b64']}", use_container_width=True)

        with c2:
            sel_id_right = st.selectbox(
                "Screenshots (rechts)",
                options=shot_df["option_id"].tolist(),
                format_func=lambda oid: label_map.get(int(oid), str(oid)),
                key="t3_shot2",
            )
            row2 = shot_df.set_index("option_id").loc[int(sel_id_right)]
            if row2.get("b64") and row2.get("mime_type"):
                st.image(f"data:{row2['mime_type']};base64,{row2['b64']}", use_container_width=True)
    else:
        st.caption("Für diese Kampagne sind noch keine Screenshots gespeichert.")

    # ------------- Daten sammeln (fused + fallbacks) -------------
    fused_latest_by_ad: dict[int, dict] = {}
    api_raw_by_ad: dict[int, dict] = {}
    for ad_id in ad_ids:
        hist = fused_history_for_ad(conn, ad_id)
        if not hist.empty:
            fused_latest = hist.tail(1).iloc[0]["fused"]
            fused_latest_by_ad[ad_id] = fused_latest
            api_raw_by_ad[ad_id] = ((fused_latest or {}).get("api") or {}).get("raw") or {}
        else:
            fused_latest_by_ad[ad_id] = {}
            api_raw_by_ad[ad_id] = {}

    # ------------- LLM-Kurzbeschreibung (Kampagne) -------------
    st.markdown("#### LLM-Kurzbeschreibung (Kampagne, 3–5 Sätze)")

    # Themen & CTA über alle Ads der Kampagne
    all_topics: list[str] = []
    for ad_id in ad_ids:
        t = fetch_topics_for_ad(conn, ad_id) or {}
        all_topics.extend([str(x) for x in (t.get("topics") or [])])
    top_topics = []
    if all_topics:
        tdf = pd.Series(all_topics).value_counts().head(5)
        top_topics = [f"{idx} (n={val})" for idx, val in tdf.items()]

    cta_vals = []
    for ad_id in ad_ids:
        llm = extract_llm_blocks(fused_latest_by_ad.get(ad_id, {})) if fused_latest_by_ad.get(ad_id) else {}
        txt = (llm or {}).get("txt") or {}
        cta_vals.append(txt.get("cta_typ") or "Unklar")
    cta_counts = pd.Series(cta_vals).value_counts().to_dict() if cta_vals else {}

    camp_prompt = f"""
Du bist Analyst:in für politische Online-Werbung in der Schweiz. Fasse die **Kampagne '{camp_name}'** über alle vorliegenden Ads prägnant in **3–5 Sätzen** zusammen – neutral, faktenbasiert.

Kontext
- Anzahl analysierter Ads: {len(ad_ids)}
- Gruppen/Akteure: {n_grp}
- Top-Themen: {", ".join(top_topics) if top_topics else "—"}
- CTA-Verteilung: {", ".join([f"{k}: {v}" for k, v in cta_counts.items()]) if cta_counts else "—"}
""".strip()

    if st.button("🔁 Kampagnen-Zusammenfassung erzeugen", key="t3_sum_btn"):
        camp_summary = None
        if _HAVE_OPENAI and os.getenv("OPENAI_API_KEY","").strip():
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.2,
                    messages=[{"role":"user","content":camp_prompt}],
                )
                camp_summary = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                st.warning(f"Automatische Kampagnen-Zusammenfassung nicht möglich ({e}).")
        st.write(camp_summary or f"{camp_name}: {len(ad_ids)} Ads; Themen: {', '.join(top_topics) if top_topics else '—'}.")

    with st.expander("Prompt (zur Nachvollziehbarkeit)"):
        st.code(camp_prompt, language="markdown")

    # ------------- Perspektiven (Kampagne) -------------
    st.subheader("Perspektiven (Kampagne)")

    # (A) campaign_perspective_results – alle page_name der Kampagne
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stance, confidence, page_name, rationale_bullets
                FROM campaign_perspective_results
                WHERE campaign_slug = %s
                """,
                (sel_camp,),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if rows else ["stance","confidence","page_name","rationale_bullets"]
        camp_persp = pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(columns=cols)
    except Exception:
        try: conn.rollback()
        except Exception: pass
        camp_persp = pd.DataFrame(columns=["stance","confidence","page_name","rationale_bullets"])

    if not camp_persp.empty:
        dist = camp_persp.groupby("stance", as_index=False).size().rename(columns={"size":"anzahl"})
        st.plotly_chart(px.bar(dist, x="stance", y="anzahl", title="Verteilung (campaign_perspective_results)"),
                        use_container_width=True)
        with st.expander("Einzel-Einträge (Kampagnen-Perspektiven)"):
            st.dataframe(
                camp_persp.assign(confidence=lambda d: d["confidence"].astype(float)).sort_values("confidence", ascending=False),
                use_container_width=True, height=260
            )
    else:
        st.caption("Keine Einträge in campaign_perspective_results gefunden.")

    # (B) Einzel-Ad Perspektiven (alle Ads der Kampagne)
    stance_df = get_perspectives_for_ads(conn, ad_ids)
    if not stance_df.empty:
        dist2 = stance_df.groupby("perspective", as_index=False).size().rename(columns={"size":"anzahl"})
        st.plotly_chart(px.bar(dist2, x="perspective", y="anzahl", title="Verteilung (Einzel-Ads)"),
                        use_container_width=True)
        with st.expander("Einzel-Ad Perspektiven"):
            st.dataframe(
                stance_df.sort_values("confidence", ascending=False, na_position="last"),
                use_container_width=True, height=260
            )
    else:
        st.caption("Keine Perspektivenklassifizierung für die Einzel-Ads gefunden.")

    # ------------- Themen (Tagging) -------------
    st.subheader("Themen (Kampagne)")
    topic_rows = []
    for ad_id in ad_ids:
        t = fetch_topics_for_ad(conn, ad_id) or {}
        for x in (t.get("topics") or []):
            topic_rows.append({"ad_id": ad_id, "topic": str(x)})
    if topic_rows:
        tdf = pd.DataFrame(topic_rows)
        top = tdf["topic"].value_counts().reset_index()
        top.columns = ["topic","anzahl"]
        st.plotly_chart(px.bar(top.head(20), x="topic", y="anzahl", title="Top-Themen (Anzahl Ads)"),
                        use_container_width=True)
        with st.expander("Themen nach Ad"):
            by_ad = tdf.groupby("ad_id")["topic"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()
            st.dataframe(by_ad, use_container_width=True, height=260)
    else:
        st.caption("Keine Themenklassifizierung vorhanden.")

    # ------------- Risiken/Schwachstellen (aggregiert) -------------
    st.subheader("Risiko-Kategorien (Ø Kampagne)")
    risk_cats = [
        "factual_accuracy","framing_quality","visual_mislead","targeting_risks",
        "policy_legal","transparency_context","consistency_history","usability_accessibility",
    ]
    risk_rows = []
    for ad_id in ad_ids:
        w = fetch_weakness_for_ad(conn, ad_id)
        if not w: continue
        payload = w.get("payload") or {}
        for k in risk_cats:
            sc = payload.get(f"score_{k}")
            if sc is not None:
                risk_rows.append({"Kategorie": k, "Score": float(sc)})

    if risk_rows:
        rdf = (pd.DataFrame(risk_rows)
               .groupby("Kategorie", as_index=False)["Score"].mean()
               .sort_values("Score", ascending=False))
        st.plotly_chart(px.bar(rdf, x="Kategorie", y="Score", title="Risiko-Kategorien (Ø)"),
                        use_container_width=True)
        st.dataframe(rdf, use_container_width=True, height=240)
    else:
        st.caption("Keine Schwachstellen-Ergebnisse gefunden.")

    # ------------- Audience & Regionen (aggregiert) -------------
    st.subheader("Audience & Regionen (aggregiert)")
    demo_parts, regs_parts = [], []
    for ad_id in ad_ids:
        api_raw = api_raw_by_ad.get(ad_id, {})
        ddf = extract_demography(api_raw)
        if not ddf.empty:
            ddf = ddf.assign(label=lambda d: d["gender"].str.title()+" "+d["age"])
            demo_parts.append(ddf[["label","spend"]])
        rdf = extract_regions(api_raw)
        if not rdf.empty:
            regs_parts.append(rdf[["region","spend"]])

    if demo_parts:
        agg_demo = (pd.concat(demo_parts, ignore_index=True)
                    .groupby("label", as_index=False)["spend"].sum()
                    .sort_values("spend", ascending=False).head(20))
        st.plotly_chart(px.bar(agg_demo, x="label", y="spend", title="Budget nach Zielgruppe (Summe)"),
                        use_container_width=True)
    else:
        st.caption("Keine demografische Verteilung verfügbar.")

    if regs_parts:
        agg_regs = (pd.concat(regs_parts, ignore_index=True)
                    .groupby("region", as_index=False)["spend"].sum()
                    .sort_values("spend", ascending=False).head(20))
        st.plotly_chart(px.bar(agg_regs, x="region", y="spend", title="Top-Regionen (Summe)"),
                        use_container_width=True)
    else:
        st.caption("Keine regionale Verteilung verfügbar.")

    # ------------- Creative-Insights (aggregiert) -------------
    st.subheader("Creative-Insights (Kampagne)")
    if cta_vals:
        cta_df = pd.Series(cta_vals, name="CTA").value_counts().reset_index()
        cta_df.columns = ["CTA","anzahl"]
        st.plotly_chart(px.bar(cta_df, x="CTA", y="anzahl", title="CTA-Verteilung"),
                        use_container_width=True)

    fl_rows = []
    comp_vals, layout_vals, platform_vals = [], [], []
    palette_vals = []
    for ad_id in ad_ids:
        llm_blocks = extract_llm_blocks(fused_latest_by_ad.get(ad_id, {})) if fused_latest_by_ad.get(ad_id) else {}
        vis = (llm_blocks or {}).get("vis") or {}
        fl = vis.get("flächenverteilung") or {}
        if fl:
            fl_rows.append({
                "Text": _num(fl.get("textfläche")),
                "Bild": _num(fl.get("bildfläche")),
                "Weißraum": _num(fl.get("weißraum")),
            })
        if vis.get("kompositionstyp"):           comp_vals.append(vis.get("kompositionstyp"))
        if vis.get("dominante_layoutstruktur"):  layout_vals.append(vis.get("dominante_layoutstruktur"))
        if vis.get("plattform"):                 platform_vals.append(vis.get("plattform"))
        for c in (vis.get("farbpalette") or []):
            c2 = _norm_hex(str(c))
            if c2: palette_vals.append(c2)

    if fl_rows:
        fl_df = pd.DataFrame(fl_rows)
        avg = fl_df.mean(numeric_only=True).reset_index()
        avg.columns = ["Typ","Anteil"]
        st.plotly_chart(px.bar(avg, x="Typ", y="Anteil", title="Flächenverteilung (Ø)"),
                        use_container_width=True)

    colsA, colsB, colsC = st.columns(3)
    if comp_vals:
        comp_df = pd.Series(comp_vals, name="Komposition").value_counts().reset_index()
        comp_df.columns = ["Komposition","anzahl"]
        with colsA:
            st.plotly_chart(px.bar(comp_df, x="Komposition", y="anzahl", title="Komposition"),
                            use_container_width=True)
    if layout_vals:
        lay_df = pd.Series(layout_vals, name="Layout").value_counts().reset_index()
        lay_df.columns = ["Layout","anzahl"]
        with colsB:
            st.plotly_chart(px.bar(lay_df, x="Layout", y="anzahl", title="Layout"),
                            use_container_width=True)
    if platform_vals:
        plat_df = pd.Series(platform_vals, name="Plattform").value_counts().reset_index()
        plat_df.columns = ["Plattform","anzahl"]
        with colsC:
            st.plotly_chart(px.bar(plat_df, x="Plattform", y="anzahl", title="Plattform (LLM)"),
                            use_container_width=True)

    if palette_vals:
        st.caption("Top-Farbpalette (Häufigkeit)")
        pal_df = pd.Series(palette_vals, name="farbe").value_counts().head(10).reset_index()
        pal_df.columns = ["farbe","anzahl"]
        cols = st.columns(len(pal_df))
        for (_, r), col in zip(pal_df.iterrows(), cols):
            col.markdown(f"<div style='height:24px;border-radius:4px;background:{r['farbe']}'></div>", unsafe_allow_html=True)
            col.caption(f"{r['farbe']} ({int(r['anzahl'])})")

    # ---------------- Zeitverlauf (Kampagne) — API-Snapshots, Summe über ALLE Ads der Kampagne ----------------
    st.subheader("Zeitverlauf (Kampagne)")

    df_camp_ads = (
        df_k[["ad_id","ad_external_id","page_name","campaign_id"]]
        .dropna(subset=["ad_id","ad_external_id"])
        .astype({"ad_id": int})
    )
    ext_ids = set(df_camp_ads["ad_external_id"].dropna().astype(str).str.strip())
    ext_ids.discard("")
    if not ext_ids:
        st.caption("Keine External-IDs für diese Kampagne gefunden.")
    else:
        try:
            campaign_ids = df_camp_ads["campaign_id"].dropna().astype(int).unique().tolist()
        except Exception:
            campaign_ids = []
        api_df = pd.DataFrame(columns=["snapshot_date","raw"])
        if campaign_ids:
            try:
                with conn.cursor() as cur:
                    cols = _table_cols(conn, "api_snapshots")
                    json_candidates = ["payload","raw","api_raw","snapshot"]
                    ts_candidates   = ["created_at","ingested_at","fetched_at","updated_at"]
                    date_col = "snapshot_date" if "snapshot_date" in cols else None
                    json_col = next((c for c in json_candidates if c in cols), None)
                    ts_col   = next((c for c in ts_candidates   if c in cols), None)
                    if json_col:
                        if date_col:
                            sql = f"""
                                SELECT {date_col} AS snapshot_date, {json_col} AS raw
                                FROM api_snapshots
                                WHERE campaign_id = ANY(%s)
                                ORDER BY {date_col}
                            """
                        else:
                            if not ts_col: ts_col = "created_at"
                            sql = f"""
                                SELECT DATE({ts_col}) AS snapshot_date, {json_col} AS raw
                                FROM api_snapshots
                                WHERE campaign_id = ANY(%s)
                                ORDER BY DATE({ts_col})
                            """
                        cur.execute(sql, (campaign_ids,))
                        rows = cur.fetchall()
                        if rows:
                            cols_out = [d[0] for d in cur.description]
                            api_df = pd.DataFrame([dict(zip(cols_out, r)) for r in rows])
            except Exception:
                try: conn.rollback()
                except Exception: pass
                api_df = pd.DataFrame(columns=["snapshot_date","raw"])

        # aggregieren (nur Tage mit mindestens 1 Kampagnen-Ad im Snapshot)
        import numpy as np
        import plotly.graph_objects as go

        ts_agg = {}
        if not api_df.empty:
            def _parse_json(x):
                if isinstance(x, str):
                    try: return json.loads(x)
                    except Exception: return []
                return x or []
            api_df["raw"] = api_df["raw"].apply(_parse_json)
            api_df["snapshot_date"] = pd.to_datetime(api_df["snapshot_date"], errors="coerce").dt.date
            api_df = api_df[api_df["snapshot_date"] != _date(2025, 9, 10)]

            for r in api_df.itertuples():
                ads_list = getattr(r, "raw", []) or []
                if not isinstance(ads_list, (list, tuple)): continue

                day_spend = day_impr = day_eu = 0.0
                day_ads = set()
                for it in ads_list:
                    ad_ext = str((it or {}).get("id") or "").strip()
                    if not ad_ext or ad_ext not in ext_ids: 
                        continue
                    day_spend += float(_mid_value((it or {}).get("spend") or {}) or 0.0)
                    day_impr  += float(_mid_value((it or {}).get("impressions") or {}) or 0.0)
                    day_eu    += float((it or {}).get("eu_total_reach") or 0)
                    day_ads.add(ad_ext)

                if day_ads:
                    ts_agg[r.snapshot_date] = {
                        "spend": day_spend,
                        "impressions": day_impr,
                        "eu_total_reach": day_eu,
                        "ads": day_ads,
                    }

        if ts_agg:
            rows = []
            for d, vals in ts_agg.items():
                rows.append({
                    "date": d,
                    "spend": vals["spend"],
                    "impressions": vals["impressions"],
                    "eu_total_reach": vals["eu_total_reach"],
                    "n_ads": len(vals["ads"]),
                    "ads_list": sorted(list(vals["ads"])),
                })
            ts_sum = pd.DataFrame(rows).sort_values("date")
        else:
            ts_sum = pd.DataFrame(columns=["date","spend","impressions","eu_total_reach","n_ads","ads_list"])

        if ts_sum.empty:
            st.caption("Keine spend/impressions/eu_total_reach in api_snapshots für diese Kampagne gefunden.")
        else:
            def _short_list(lst, n=6):
                if not lst: return "—"
                return ", ".join(lst) if len(lst) <= n else ", ".join(lst[:n]) + f" … (+{len(lst)-n})"
            ts_sum["ads_short"] = ts_sum["ads_list"].apply(_short_list)

            # Spend
            fig_spend = go.Figure()
            fig_spend.add_trace(go.Scatter(
                x=ts_sum["date"], y=ts_sum["spend"], mode="lines+markers", name="Spend",
                customdata=np.stack([ts_sum["n_ads"], ts_sum["ads_short"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>Spend (Summe): %{y:.0f} CHF"
                              "<br>Beiträge von %{customdata[0]} Ads"
                              "<br><span style='font-size:11px'>%{customdata[1]}</span>"
                              "<extra></extra>"
            ))
            fig_spend.update_layout(
                title="Spend über Zeit (API-Snapshots; Summe Kampagne)",
                xaxis_title="Datum", yaxis_title="Spend (CHF)",
                showlegend=False
            )
            st.plotly_chart(fig_spend, use_container_width=True)

            # Impressions
            fig_impr = go.Figure()
            fig_impr.add_trace(go.Scatter(
                x=ts_sum["date"], y=ts_sum["impressions"], mode="lines+markers", name="Impressions",
                customdata=np.stack([ts_sum["n_ads"], ts_sum["ads_short"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>Impressions (Summe): %{y:.0f}"
                              "<br>Beiträge von %{customdata[0]} Ads"
                              "<br><span style='font-size:11px'>%{customdata[1]}</span>"
                              "<extra></extra>"
            ))
            fig_impr.update_layout(
                title="Impressions über Zeit (API-Snapshots; Summe Kampagne)",
                xaxis_title="Datum", yaxis_title="Impressions",
                showlegend=False
            )
            st.plotly_chart(fig_impr, use_container_width=True)

            # EU-Reach
            fig_eu = go.Figure()
            fig_eu.add_trace(go.Scatter(
                x=ts_sum["date"], y=ts_sum["eu_total_reach"], mode="lines+markers", name="EU-Reach",
                customdata=np.stack([ts_sum["n_ads"], ts_sum["ads_short"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>EU-Reach (Summe): %{y:.0f}"
                              "<br>Beiträge von %{customdata[0]} Ads"
                              "<br><span style='font-size:11px'>%{customdata[1]}</span>"
                              "<extra></extra>"
            ))
            fig_eu.update_layout(
                title="EU-Reach über Zeit (API-Snapshots; Summe Kampagne)",
                xaxis_title="Datum", yaxis_title="EU-Reach",
                showlegend=False
            )
            st.plotly_chart(fig_eu, use_container_width=True)

            # Vollständige Liste (Validierung)
            st.markdown("**Verwendete Ads (in den Snapshots gefunden):**")
            used_all_ext = sorted(set(e for lst in ts_sum["ads_list"] for e in lst))
            if used_all_ext:
                meta_idx = df_camp_ads.set_index("ad_external_id")
                bullets = []
                for ext in used_all_ext:
                    if ext in meta_idx.index:
                        row = meta_idx.loc[ext]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        label = f"{row['page_name']} ({ext}) — Ad {int(row['ad_id'])}"
                    else:
                        label = f"{ext}"
                    bullets.append(f"- {label}")
                st.markdown("\n".join(bullets))
            else:
                st.caption("Keine passenden Ads in den Snapshots gefunden.")
