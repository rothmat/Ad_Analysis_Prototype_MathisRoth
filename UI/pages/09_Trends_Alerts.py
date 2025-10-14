#pages/09_Trends_Alerts
# -*- coding: utf-8 -*-
import json, math, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

st.set_page_config(page_title="Trends & Alerts", page_icon="📈", layout="wide")
st.title("📈 Trends & Alerts")

# ---------------- DB helpers ----------------

def get_campaign_options(conn) -> List[Tuple[int, str, str]]:
    sql = "SELECT id, name, slug FROM campaigns ORDER BY name"
    with conn.cursor() as cur:
        cur.execute(sql)
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

def get_llm_ads_for_campaign(conn, campaign_slug: str) -> pd.DataFrame:
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
    return pd.DataFrame([{"ad_pk": r[0], "ad_external_id": r[1]} for r in rows])

def load_fused_rows(conn, campaign_slug: str, ad_pks: List[int]) -> pd.DataFrame:
    if not ad_pks:
        return pd.DataFrame(columns=["ad_pk","snapshot_date","fused"])
    sql = """
      SELECT f.ad_id AS ad_pk, f.snapshot_date, f.fused, f.created_at
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
    for ad_pk, snapshot_date, fused, _ in rows:
        if isinstance(fused, str):
            try:
                fused = json.loads(fused)
            except Exception:
                fused = {}
        out.append({"ad_pk": int(ad_pk), "snapshot_date": str(snapshot_date), "fused": fused})
    return pd.DataFrame(out)

# ---------------- Utilities / Extraction ----------------

def _num(x) -> float:
    try:
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"): return float(s[:-1]) / 100.0
            return float(s)
        return float(x)
    except Exception:
        return float("nan")

def mid_from_bounds(bounds: Dict[str, Any]) -> float:
    lo = _num((bounds or {}).get("lower_bound"))
    hi = _num((bounds or {}).get("upper_bound"))
    if math.isnan(lo) and math.isnan(hi):
        return float("nan")
    if math.isnan(lo): lo = hi
    if math.isnan(hi): hi = lo
    return float((lo + hi) / 2.0)

def extract_regions(api_raw: Dict[str,Any]) -> pd.DataFrame:
    reg = (api_raw.get("delivery_by_region")
           or api_raw.get("region_distribution")
           or api_raw.get("region_breakdown")
           or api_raw.get("regions"))
    rows=[]
    if not reg: return pd.DataFrame(rows)
    if isinstance(reg, list):
        for r in reg:
            region = r.get("region") or r.get("name") or r.get("key")
            share = _num(r.get("percentage") or r.get("share") or r.get("value"))
            rows.append({"region": str(region), "share": share})
    elif isinstance(reg, dict):
        for region, val in reg.items():
            rows.append({"region": str(region), "share": _num(val)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["share"] = df["share"].apply(lambda v: v/100.0 if (pd.notna(v) and v > 1.0) else v).fillna(0.0)
    return df

# ---- Labels & Meta (wie auf „Audience & Regionen“) ----
def _media_id_from_raw(raw: Dict[str,Any]) -> str | None:
    if not isinstance(raw, dict): return None
    cand = raw.get("media_id") or raw.get("id")
    if cand: return str(cand)
    url = str(raw.get("ad_snapshot_url") or "")
    m = re.search(r"[?&]id=(\d+)", url)
    return m.group(1) if m else None

def _extract_group(api_raw: Dict[str,Any]) -> str:
    g = str((api_raw or {}).get("page_name") or "").strip()
    return g if g else "(keine Angabe)"

def _extract_sponsors(api_raw: Dict[str,Any]) -> List[str]:
    raw = (api_raw or {}).get("bylines")
    out: List[str] = []
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        out = [s.strip() for s in re.split(r"[|,;/]+", raw) if s.strip()]
    return out or ["(keine Angabe)"]

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

ads_df = get_llm_ads_for_campaign(conn, campaign_slug)

# ---- Meta ziehen, um Labels „Gruppe (media_id)“ zu bauen
all_ad_pks = [int(r.ad_pk) for r in ads_df.itertuples()]
_meta_rows = load_fused_rows(conn, campaign_slug, all_ad_pks)

latest_meta: Dict[int, Dict[str,str | None]] = {}
if not _meta_rows.empty:
    for ad_pk, grp in _meta_rows.groupby("ad_pk", sort=False):
        last = grp.tail(1).iloc[0]
        api_raw = (last["fused"].get("api") or {}).get("raw") or {}
        latest_meta[int(ad_pk)] = {
            "page_name": (_extract_group(api_raw) or None),
            "media_id":  (_media_id_from_raw(api_raw) or None)
        }

def _label_for_ad(ad_pk: int, ad_external_id: Any) -> str:
    meta = latest_meta.get(int(ad_pk), {})
    pn = meta.get("page_name") or "-"
    mid = meta.get("media_id") or (str(ad_external_id) if ad_external_id is not None else "-")
    # final format: Gruppe (media_id)
    return f"{pn} ({mid})" if pn and mid else f"Ad {ad_pk}"

ad_options = [_label_for_ad(int(r.ad_pk), r.ad_external_id) for r in ads_df.itertuples()]
label_to_pk = {_label_for_ad(int(r.ad_pk), r.ad_external_id): int(r.ad_pk) for r in ads_df.itertuples()}

# ---- Auswahl + „Alle auswählen“ (robust)
SEL_KEY  = "sel_ads_09"
ALL_FLAG = "__sel_ads_09_all_flag__"

# 1) Flag auswerten, BEVOR das Widget gerendert wird
if st.session_state.get(ALL_FLAG):
    st.session_state[SEL_KEY] = ad_options[:]   # komplette Liste
    st.session_state.pop(ALL_FLAG, None)

col_sel, col_btn = st.columns([5,1])
with col_sel:
    prev = st.session_state.get(SEL_KEY, ad_options[:min(6, len(ad_options))])
    default_vals = [v for v in prev if v in ad_options] or ad_options[:min(6, len(ad_options))]
    sel_labels = st.multiselect("Ads auswählen", options=ad_options, default=default_vals, key=SEL_KEY)

with col_btn:
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    if st.button("Alle auswählen", use_container_width=True, key="btn_all_ads_09"):
        st.session_state[ALL_FLAG] = True
        st.rerun()

selected_ad_pks = [label_to_pk[l] for l in st.session_state.get(SEL_KEY, []) if l in label_to_pk]

st.divider()

if not selected_ad_pks:
    st.info("Bitte mindestens eine Ad auswählen.")
    st.stop()

fused_df = load_fused_rows(conn, campaign_slug, selected_ad_pks)
if fused_df.empty:
    st.info("Keine fused-Daten für die Auswahl."); st.stop()

# ---------------- Optionale Filter: Gruppe & Sponsoren (wie Audience & Regionen) ----------------
st.subheader("Filter")
c0, c1, c2 = st.columns([1,3,3])
with c0:
    enable_group = st.checkbox("Gruppe filtern", value=False)
    enable_spon  = st.checkbox("Sponsoren filtern", value=False)

# vorbereitende Spalten für Filter
fused_df["__api_raw"] = fused_df["fused"].apply(lambda f: (f.get("api") or {}).get("raw") or {})
fused_df["__group"]   = fused_df["__api_raw"].apply(_extract_group)
fused_df["__sponsors"]= fused_df["__api_raw"].apply(_extract_sponsors)

all_groups   = sorted(fused_df["__group"].unique()) if enable_group else []
all_sponsors = sorted({s for lst in fused_df["__sponsors"] for s in lst}) if enable_spon else []

with c1:
    sel_groups = st.multiselect(
        "Gruppe (page_name)",
        options=all_groups if all_groups else [],
        default=all_groups if all_groups else [],
        disabled=not enable_group,
    )
with c2:
    sel_sponsors = st.multiselect(
        "Sponsoren (bylines)",
        options=all_sponsors if all_sponsors else [],
        default=all_sponsors if all_sponsors else [],
        disabled=not enable_spon,
    )

# Bool-Maske IMMER als Series starten
mask = pd.Series(True, index=fused_df.index)
if enable_group and sel_groups:
    mask &= fused_df["__group"].isin(set(sel_groups))
if enable_spon and sel_sponsors:
    sset = set(sel_sponsors)
    mask &= fused_df["__sponsors"].apply(lambda lst: any(s in sset for s in lst))

filtered = fused_df[mask].copy()
if filtered.empty:
    st.info("Keine Daten nach den Filtern."); st.stop()

# ---------------- Spend/Impressions-Zeitreihe (gefiltert) ----------------
ts_rows=[]
for row in filtered.itertuples():
    api = (row.fused.get("api") or {}).get("raw") or {}
    spend = mid_from_bounds(api.get("spend") or {})
    impr  = mid_from_bounds(api.get("impressions") or {})
    # wichtig: Datum als echtes Tagesdatum vereinheitlichen
    day = pd.to_datetime(row.snapshot_date).date()
    ts_rows.append({"date": day, "ad_pk": row.ad_pk,
                    "spend": 0.0 if math.isnan(spend) else spend,
                    "impressions": 0.0 if math.isnan(impr) else impr})
ts = pd.DataFrame(ts_rows)

if not ts.empty:
    st.plotly_chart(px.line(ts.groupby("date", as_index=False)["spend"].sum(), x="date", y="spend",
                            title="Ausgaben über Zeit"), use_container_width=True)
    st.plotly_chart(px.area(ts.groupby("date", as_index=False)["impressions"].sum(), x="date", y="impressions",
                            title="Impressions über Zeit"), use_container_width=True)

# ---------------- CTA-Mix über Zeit (gefiltert) ----------------
cta_rows=[]
for row in filtered.itertuples():
    llm = ((row.fused.get("llm_analysis") or {}).get("analysis_file_payload") or {})
    txt = (llm.get("analyse") or {}).get("textuelle_features") or {}
    cta = txt.get("cta_typ") or "Unklar"
    day = pd.to_datetime(row.snapshot_date).date()
    cta_rows.append({"date": day, "CTA": cta, "value": 1})
ctats = pd.DataFrame(cta_rows)
if not ctats.empty:
    mix = ctats.groupby(["date","CTA"], as_index=False)["value"].sum()
    st.plotly_chart(px.area(mix, x="date", y="value", color="CTA", title="CTA-Mix über Zeit"),
                    use_container_width=True)

# ---------------- Region-Drift (verbessert, normierte JS-Divergenz 0–1) ----------------
# Erklärung für Nutzende
with st.expander("Was bedeutet „Region-Drift“?"):
    st.markdown(
        "- Wir vergleichen die heutige regionale Verteilung der Ausspielung mit einer **Rolling-Baseline** "
        "(z. B. letzte 7 Tage, spend-gewichtet). "
        "Die **JS-Divergenz** misst den Unterschied zweier Verteilungen. "
        "Wir **normieren** sie auf **0–1**: 0 = identisch, 1 = maximal verschieden. "
        "Nur wenn genügend Spend vorliegt, melden wir eine Drift und zeigen die wichtigsten **Treiber (Δ in Prozentpunkten)**."
    )

# ---- Sidebar: erweiterte Einstellungen für Drift
with st.sidebar:
    st.header("Alert-Regeln")
    win      = st.number_input("Rolling-Fenster (Tage)", 2, 14, 5)
    ratio    = st.number_input("Spike-Ratio (x)", 1.1, 5.0, 1.4, step=0.1)
    base_win = st.number_input("Baseline für Region-Drift (Tage)", 2, 14, 7)
    js_thr   = st.number_input("Region-Drift-Schwelle (normiert 0–1)", 0.0, 1.0, 0.25, step=0.05, format="%.2f")
    min_spend_for_drift = st.number_input("Mindest-Spend/Tag für Drift (Midpoint)", 0.0, 1e9, 50.0, step=10.0)

# Hilfen für Drift-Berechnung
LN2 = math.log(2.0)
def js_norm(p: pd.Series, q: pd.Series) -> float:
    p = (p.fillna(0)); q = (q.fillna(0))
    ps = p / (p.sum() or 1); qs = q / (q.sum() or 1)
    m  = 0.5 * (ps + qs); eps = 1e-12
    kl = lambda a,b: float((a * np.log((a+eps)/(b+eps))).sum())
    return (0.5 * kl(ps, m) + 0.5 * kl(qs, m)) / LN2  # normiert auf 0..1

# Tages-Verteilungen sammeln
daily_dists: dict[str, List[pd.Series]] = {}
for row in filtered.itertuples():
    api = (row.fused.get("api") or {}).get("raw") or {}
    reg = extract_regions(api)
    if reg.empty:
        continue
    day_key = pd.to_datetime(row.snapshot_date).date().isoformat()
    dist = reg.groupby("region")["share"].sum()
    daily_dists.setdefault(day_key, []).append(dist)

# Tages-Spend (für Guards + gewichtete Baseline)
daily_spend_series = ts.groupby("date", as_index=False)["spend"].sum()
daily_spend = {d.date().isoformat(): float(s) for d, s in zip(pd.to_datetime(daily_spend_series["date"]), daily_spend_series["spend"])}

def _mix(series_list: List[pd.Series]) -> pd.Series:
    s = None
    for ser in series_list:
        s = ser if s is None else s.add(ser, fill_value=0)
    s = s if s is not None else pd.Series(dtype=float)
    return s / (s.sum() or 1)

dates_sorted = sorted(daily_dists.keys())
drift_rows = []
for i, d in enumerate(dates_sorted):
    # aktuelle Tagesverteilung
    cur = _mix(daily_dists[d])

    # Baseline: vorherige base_win Tage, spend-gewichtet
    d_ts = pd.to_datetime(d)
    prev_dates = [dd for dd in dates_sorted[:i] if (d_ts - pd.to_datetime(dd)).days <= int(base_win)]
    if not prev_dates:
        continue
    baseline = None; wsum = 0.0
    for dd in prev_dates:
        dist_dd = _mix(daily_dists[dd])
        w = daily_spend.get(dd, 0.0)
        baseline = dist_dd.mul(w) if baseline is None else baseline.add(dist_dd.mul(w), fill_value=0)
        wsum += w
    if baseline is None or wsum <= 0:
        continue
    baseline = baseline / wsum

    # Guards
    if daily_spend.get(d, 0.0) < float(min_spend_for_drift):
        continue
    if len(cur.dropna()) < 2 or len(baseline.dropna()) < 2:
        continue

    # normierte JS-Divergenz
    jsn = js_norm(baseline, cur)

    # Treiber (Δ in Prozentpunkten ggü. Baseline)
    delta = (cur - baseline).sort_values(ascending=False)
    top_up   = delta.head(2)
    top_down = delta.tail(2)

    drift_rows.append({
        "date": pd.to_datetime(d).date(),
        "js_norm": jsn,
        "top_up":   ", ".join([f"{k} {v*100:+.1f}pp" for k, v in top_up.items()]) if not top_up.empty else "",
        "top_down": ", ".join([f"{k} {v*100:+.1f}pp" for k, v in top_down.items()]) if not top_down.empty else "",
    })

drift = pd.DataFrame(drift_rows).sort_values("date")

# Plot + Alerts
if not drift.empty:
    fig = px.line(drift, x="date", y="js_norm", title="Region Drift (normierte JS-Divergenz, 0–1)")
    fig.add_hline(y=float(js_thr), line_dash="dash",
                  annotation_text=f"Schwelle {float(js_thr):.2f}",
                  annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    def _sev(x: float) -> str:
        return "leicht" if x < 0.15 else ("mittel" if x < 0.35 else "stark")

# --- Alerts (Spend-Spikes + Drift) ---
with st.sidebar:
    st.subheader("Ergebnisse")

# Spend-Spikes (wie gehabt)
df_alert = ts.groupby("date", as_index=False)["spend"].sum().sort_values("date")
df_alert["roll"] = df_alert["spend"].rolling(int(win)).mean()
df_alert["spike"] = (df_alert["spend"] > (df_alert["roll"] * float(ratio))).fillna(False)

spikes = df_alert[df_alert["spike"]]
if not spikes.empty:
    for _, r in spikes.iterrows():
        st.success(f"Spike **{r['date']}** – Spend {r['spend']:.0f} (> {ratio}× Rolling).")
else:
    st.caption("Keine Spend-Spikes gemäß Heuristik entdeckt.")

# Drift-Alerts mit verständlicher Beschreibung
if not drift.empty and (drift["js_norm"] > float(js_thr)).any():
    for _, r in drift[drift["js_norm"] > float(js_thr)].iterrows():
        sev = _sev(float(r["js_norm"]))
        st.warning(
            f"**Region-Drift {r['date']} – {sev}** "
            f"(JS {float(r['js_norm']):.2f} > {float(js_thr):.2f}). "
            f"Treiber: ▲ {r['top_up'] or '–'}; ▼ {r['top_down'] or '–'}."
        )
else:
    st.caption("Kein Region-Drift über der Schwelle.")
