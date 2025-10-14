#05_Audience_Regionen
# -*- coding: utf-8 -*-
import json, math, unicodedata, re, warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

# Warnungen zu pandas observed=False unterdrücken
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*observed=False is deprecated.*",
)

# DB client
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

st.set_page_config(page_title="Audience & Regionen", page_icon="🧑‍🤝‍🧑", layout="wide")
st.title("🧑‍🤝‍🧑 Audience & Regionen")

# ---------------------------------------------------------------------
# Verlässliche GeoJSON-Quelle (CH Kantone) — wird bei Bedarf automatisch geladen
# ---------------------------------------------------------------------
DEFAULT_CH_GEOJSON_URL = (
    "https://public.opendatasoft.com/explore/dataset/"
    "georef-switzerland-kanton/download/?format=geojson&timezone=Europe%2FBerlin&lang=en"
)

# Offizielle Kantonskürzel -> kanonischer Name (klein geschrieben)
CODES = {
    "ZH": "zürich", "BE": "bern", "LU": "luzern", "UR": "uri", "SZ": "schwyz", "OW": "obwalden",
    "NW": "nidwalden", "GL": "glarus", "ZG": "zug", "FR": "fribourg", "SO": "solothurn",
    "BS": "basel-stadt", "BL": "basel-landschaft", "SH": "schaffhausen",
    "AR": "appenzell ausserrhoden", "AI": "appenzell innerrhoden", "SG": "st. gallen",
    "GR": "graubünden", "AG": "aargau", "TG": "thurgau", "TI": "ticino", "VD": "vaud",
    "VS": "valais", "NE": "neuchâtel", "GE": "genève", "JU": "jura",
}
CH_CANTON_CENTROIDS = {
    "ZH": (47.3769, 8.5417),   "BE": (46.9481, 7.4474),   "LU": (47.0502, 8.3093),
    "UR": (46.8803, 8.6370),   "SZ": (47.0207, 8.6541),   "OW": (46.8963, 8.2473),
    "NW": (46.9570, 8.3653),   "GL": (47.0406, 9.0680),   "ZG": (47.1662, 8.5155),
    "FR": (46.8065, 7.1619),   "SO": (47.2079, 7.5371),   "BS": (47.5596, 7.5886),
    "BL": (47.4840, 7.7366),   "SH": (47.6973, 8.6349),   "AR": (47.3833, 9.2833),
    "AI": (47.3310, 9.4096),   "SG": (47.4245, 9.3767),   "GR": (46.8508, 9.5328),
    "AG": (47.3904, 8.0457),   "TG": (47.5581, 8.8980),   "TI": (46.1950, 9.0296),
    "VD": (46.5197, 6.6323),   "VS": (46.2330, 7.3606),   "NE": (46.9899, 6.9293),
    "GE": (46.2044, 6.1432),   "JU": (47.3661, 7.3445),
}
CODE_SET = set(CODES.keys())

# ----------------------------- DB helpers -----------------------------
def get_campaign_options(conn) -> List[Tuple[int, str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, slug FROM campaigns ORDER BY name")
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

# ----------------------------- Extractors -----------------------------
def _num(x) -> float:
    try:
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"):
                return float(s[:-1]) / 100.0
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

def extract_demography(api_raw: Dict[str,Any]) -> pd.DataFrame:
    rows = []
    dem = api_raw.get("demographic_distribution") or []
    spend_mid = mid_from_bounds(api_raw.get("spend") or {})
    if isinstance(dem, dict):
        for k, v in dem.items():
            parts = str(k).replace("|"," ").replace(","," ").split()
            gender = next((p for p in parts if p.lower() in ("male","female","unknown")), "unknown")
            age = next((p for p in parts if "-" in p or p.endswith("+")), "unknown")
            share = _num(v)
            rows.append({"age": age, "gender": gender, "share": share,
                         "spend": (spend_mid or 0) * (share or 0)})
    elif isinstance(dem, list):
        for d in dem:
            age = d.get("age")
            gender = d.get("gender")
            share = _num(d.get("percentage"))
            rows.append({"age": age, "gender": gender, "share": share,
                         "spend": (spend_mid or 0) * (share or 0)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["share"] = df["share"].apply(lambda v: v/100.0 if (pd.notna(v) and v > 1.0) else v).fillna(0.0)
        df["spend"] = df["spend"].fillna(0.0)
    return df

def extract_regions(api_raw: Dict[str,Any]) -> pd.DataFrame:
    reg = (api_raw.get("delivery_by_region")
           or api_raw.get("region_distribution")
           or api_raw.get("region_breakdown")
           or api_raw.get("regions"))
    rows = []
    if not reg:
        return pd.DataFrame(rows)
    spend_mid = mid_from_bounds(api_raw.get("spend") or {})
    if isinstance(reg, list):
        for r in reg:
            region = r.get("region") or r.get("name") or r.get("key")
            share = _num(r.get("percentage") or r.get("share") or r.get("value"))
            rows.append({"region_raw": str(region), "share": share,
                         "spend": (spend_mid or 0) * (share or 0)})
    elif isinstance(reg, dict):
        for region, val in reg.items():
            rows.append({"region_raw": str(region), "share": _num(val),
                         "spend": (spend_mid or 0) * (_num(val) or 0)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["share"] = df["share"].apply(lambda v: v/100.0 if (pd.notna(v) and v > 1.0) else v).fillna(0.0)
        df["spend"] = df["spend"].fillna(0.0)
    return df

def extract_publisher_platforms(api_raw: Dict[str,Any]) -> pd.DataFrame:
    plats = api_raw.get("publisher_platforms") or []
    if isinstance(plats, str):
        plats = [plats]
    spend_mid = mid_from_bounds(api_raw.get("spend") or {})
    rows = [{"platform": str(p).lower(), "spend": (spend_mid or 0)} for p in plats]
    return pd.DataFrame(rows)

# ------------------------- Normalisierung / Mapping --------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().replace("’","'").replace("`","'").strip()

def _clean_tokens(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"\b(canton|kanton|of|de|du|des|la|le|les)\b", " ", s)
    s = re.sub(r"[^a-z0-9+ .-]", " ", s)
    s = (s.replace("wallis","valais")
           .replace("freiburg","fribourg")
           .replace("geneva","genève")
           .replace("basel city","basel-stadt")
           .replace("basel-city","basel-stadt")
           .replace("grisons","graubunden"))
    s = re.sub(r"\s+", " ", s).strip()
    return s

CODES_REV = {_clean_tokens(v): k for k, v in CODES.items()}

SYNONYMS: Dict[str, List[str]] = {
    "Zürich": ["zurich","zuerich","zuri"],
    "Bern": ["berne"],
    "Luzern": ["lucerne"],
    "Fribourg": ["freiburg"],
    "Basel-Stadt": ["basel stadt","basel-city","basel city","basel"],
    "Basel-Landschaft": ["basel land"],
    "St. Gallen": ["sankt gallen","saint gall","st gallen","st.gallen"],
    "Graubünden": ["graubunden","grigioni","grisons","graubuenden"],
    "Ticino": ["tessin"],
    "Vaud": ["waadt"],
    "Valais": ["wallis"],
    "Neuchâtel": ["neuenburg","neuchatel"],
    "Genève": ["geneva","geneve","genf"],
}

def _name_to_code(name: str) -> Optional[str]:
    raw = _clean_tokens(name)
    for code, cname in CODES.items():
        if raw == _clean_tokens(cname):
            return code
    if len(raw) == 2 and raw.upper() in CODES:
        return raw.upper()
    for canonical, syns in SYNONYMS.items():
        if raw in {_clean_tokens(s) for s in syns}:
            code = CODES_REV.get(_clean_tokens(canonical))
            if code:
                return code
    m = re.match(r"(canton\s+of\s+)?(.+)", raw)
    if m:
        cand = _clean_tokens(m.group(2))
        for code, cname in CODES.items():
            if cand == _clean_tokens(cname):
                return code
    return None

# ------------------------- Swiss map helpers --------------------------
@st.cache_data(show_spinner=False)
def _load_swiss_geojson() -> Dict[str,Any] | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "core" / "data" / "ch_cantons.geojson",
        Path(__file__).resolve().parents[2] / ".run" / "ch_cantons.geojson",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    # Auto-Download (falls möglich)
    target = candidates[1]
    try:
        import requests
        target.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(DEFAULT_CH_GEOJSON_URL, timeout=20)
        if r.ok and r.text.strip().startswith("{"):
            target.write_text(r.text, encoding="utf-8")
            return json.loads(r.text)
    except Exception:
        pass
    return None

def _geojson_add_ids_and_index(geo: Dict[str,Any]) -> Tuple[Dict[str,Any], Dict[str,str]]:
    feats = (geo or {}).get("features") or []
    index: Dict[str,str] = {}
    for f in feats:
        props = f.get("properties") or {}
        candidates = []
        for k in ["name","NAME","kanton","KANTONSNAME","kan_name","nom","abbr","abbrev","code","ktn","ktn_code","kantonskuerzel","kantonskürzel"]:
            v = props.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
        code = None
        for v in candidates:
            if len(v) == 2 and v.upper() in CODES:
                code = v.upper(); break
        if code is None:
            for v in candidates:
                code = _name_to_code(v)
                if code: break
        if code is None and "canton" in props and isinstance(props["canton"], str):
            code = _name_to_code(props["canton"])
        if not code:
            continue
        f["id"] = code
        for v in candidates:
            index[_clean_tokens(v)] = code
        index[code] = code
        index[code.lower()] = code
        index[_clean_tokens(CODES[code])] = code
    return geo, index

# ------------------------------ UI flow ------------------------------
try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")
    st.stop()

camp_opts = get_campaign_options(conn)
if not camp_opts:
    st.info("Keine Kampagnen vorhanden."); st.stop()

labels = [f"{n} ({s})" for _, n, s in camp_opts]
idx_sel = st.selectbox("Kampagne", options=list(range(len(camp_opts))), format_func=lambda i: labels[i])
campaign_id, campaign_name, campaign_slug = camp_opts[idx_sel]

# ---- Ad-Liste mit schöneren Labels: "<page_name> (<media_id>)"
ads_df = get_llm_ads_for_campaign(conn, campaign_slug)

# alle Ads einmalig laden, um page_name & media_id zu finden
def _media_id_from_raw(raw: Dict[str,Any]) -> Optional[str]:
    if not isinstance(raw, dict): return None
    cand = raw.get("media_id") or raw.get("id")
    if cand: return str(cand)
    url = str(raw.get("ad_snapshot_url") or "")
    m = re.search(r"[?&]id=(\d+)", url)
    return m.group(1) if m else None

all_ad_pks = [int(r.ad_pk) for r in ads_df.itertuples()]
meta_rows = load_fused_rows(conn, campaign_slug, all_ad_pks)

latest_meta = {}
if not meta_rows.empty:
    # unsere load_fused_rows kommt schon nach Datum sortiert -> tail(1) pro Ad
    for ad_pk, grp in meta_rows.groupby("ad_pk", sort=False):
        last = grp.tail(1).iloc[0]
        api_raw = (last["fused"].get("api") or {}).get("raw") or {}
        page_name = str(api_raw.get("page_name") or "").strip()
        media_id  = _media_id_from_raw(api_raw)
        latest_meta[int(ad_pk)] = {
            "page_name": page_name or None,
            "media_id":  media_id or None
        }

def _label_for_ad(ad_pk: int, ad_external_id: Any) -> str:
    meta = latest_meta.get(int(ad_pk), {})
    pn = meta.get("page_name") or "-"
    mid = meta.get("media_id") or (str(ad_external_id) if ad_external_id is not None else "-")
    return f"{pn} ({mid})"

ad_options = [
    _label_for_ad(int(r.ad_pk), r.ad_external_id) for r in ads_df.itertuples()
]
label_to_pk = {
    _label_for_ad(int(r.ad_pk), r.ad_external_id): int(r.ad_pk) for r in ads_df.itertuples()
}

col_sel, col_btn = st.columns([5,1])
with col_sel:
    default_sel = st.session_state.get("sel_ads_05", ad_options[:min(6, len(ad_options))])
    sel_labels = st.multiselect("Ads auswählen", options=ad_options, default=default_sel)
with col_btn:
    # Spacer wie in „Creative Insights“, damit Button auf Höhe des Eingabefeldes sitzt
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    if st.button("Alle auswählen", use_container_width=True):
        st.session_state["sel_ads_05"] = ad_options[:]
        st.rerun()

selected_ad_pks = [label_to_pk[l] for l in sel_labels if l in label_to_pk]

st.divider()

if not selected_ad_pks:
    st.info("Bitte mindestens eine Ad auswählen.")
    st.stop()

fused_df = load_fused_rows(conn, campaign_slug, selected_ad_pks)
if fused_df.empty:
    st.info("Keine fused-Daten für die Auswahl."); st.stop()

# ------------------------- Optionale Filter: Gruppe & Sponsoren -------------------------
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

st.subheader("Filter")
c0, c1, c2 = st.columns([1,3,3])
with c0:
    enable_group = st.checkbox("Gruppe filtern", value=False)
    enable_spon  = st.checkbox("Sponsoren filtern", value=False)

all_groups: List[str] = []
all_sponsors: List[str] = []
if enable_group or enable_spon:
    gset, sset = set(), set()
    for row in fused_df.itertuples():
        api_raw = (row.fused.get("api") or {}).get("raw") or {}
        if enable_group:
            gset.add(_extract_group(api_raw))
        if enable_spon:
            for s in _extract_sponsors(api_raw):
                sset.add(s)
    all_groups = sorted(gset)
    all_sponsors = sorted(sset)

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

sel_groups_set = set(sel_groups)
sel_sponsors_set = set(sel_sponsors)

def _row_passes_filters(api_raw: Dict[str,Any]) -> bool:
    if enable_group:
        g = _extract_group(api_raw)
        if g not in sel_groups_set:
            return False
    if enable_spon:
        sponsors = set(_extract_sponsors(api_raw))
        if not (sponsors & sel_sponsors_set):
            return False
    return True

# ------------------------- Audience / Plattformen -------------------------
demo_rows, plat_rows = [], []
for row in fused_df.itertuples():
    api_raw = (row.fused.get("api") or {}).get("raw") or {}
    if not _row_passes_filters(api_raw):
        continue
    ddf = extract_demography(api_raw)
    if not ddf.empty:
        demo_rows.append(ddf)
    pdf = extract_publisher_platforms(api_raw)
    if not pdf.empty:
        plat_rows.append(pdf)

demo = pd.concat(demo_rows, ignore_index=True) if demo_rows else pd.DataFrame()
plats = pd.concat(plat_rows, ignore_index=True) if plat_rows else pd.DataFrame()

# Budget nach Zielgruppe
if not demo.empty:
    agg_aud = (demo.assign(label=lambda d: d["gender"].str.title() + " " + d["age"])
                    .groupby("label", as_index=False, observed=False)["spend"].sum()
                    .sort_values("spend", ascending=False))
    fig_aud = px.bar(agg_aud, x="label", y="spend", title="Budget nach Zielgruppe (mid)")
    fig_aud.update_layout(xaxis_title=None, yaxis_title="Spend (mid)", height=420)
    st.plotly_chart(fig_aud, use_container_width=True)
else:
    st.caption("Keine Audience-Verteilung (nach aktuellen Filtern).")

# Demografie-Heatmap
if not demo.empty:
    st.subheader("Demografie-Heatmap (Spend)")
    age_order = ["18-24","25-34","35-44","45-54","55-64","65+"]
    gender_order = ["female","male","unknown"]
    demo["age"] = pd.Categorical(demo["age"], categories=age_order, ordered=True)
    demo["gender"] = pd.Categorical(demo["gender"], categories=gender_order, ordered=True)
    heat = (demo.groupby(["age","gender"], as_index=False, observed=False)["spend"].sum()
                 .pivot(index="age", columns="gender", values="spend")
                 .reindex(index=age_order, columns=gender_order))
    fig_hm = px.imshow(
        heat,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(color="Spend (mid)"),
    )
    fig_hm.update_layout(
        xaxis_title="Geschlecht",
        yaxis_title="Altersgruppe",
        coloraxis_colorbar=dict(title="Spend"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# Publisher-Plattformen
if not plats.empty:
    agg_plat = (plats.groupby("platform", as_index=False, observed=False)["spend"].sum()
                     .sort_values("spend", ascending=False))
    st.subheader("Publisher-Plattformen (budget-gewichtet)")
    st.plotly_chart(px.bar(agg_plat, x="platform", y="spend", title=None, height=420),
                    use_container_width=True)

# ------------------------- Regionen (CH) -------------------------
# Daten sammeln – nach Filtern
reg_rows = []
for row in fused_df.itertuples():
    api_raw = (row.fused.get("api") or {}).get("raw") or {}
    if not _row_passes_filters(api_raw):
        continue
    rdf = extract_regions(api_raw)
    if not rdf.empty:
        reg_rows.append(rdf)

regions = pd.concat(reg_rows, ignore_index=True) if reg_rows else pd.DataFrame()

if regions.empty:
    st.caption("Keine regionale Verteilung (nach aktuellen Filtern).")
else:
    st.subheader("Regionale Heatmap (Schweiz, budget-gewichtet)")

    # Aggregation zu Kantonscodes
    mapped = []
    for r in regions.itertuples():
        code = _name_to_code(r.region_raw)
        if code:
            mapped.append({"code": code, "spend": float(r.spend)})
    agg_spend = pd.DataFrame(mapped).groupby("code", as_index=False)["spend"].sum() \
                if mapped else pd.DataFrame(columns=["code","spend"])

    # Bubble-Heatmap (Marker dezenter, Hintergrund deutlicher)
    rows_map = []
    total_spend = float(agg_spend["spend"].sum() or 0)
    w_lat = w_lon = 0.0
    for _, r in agg_spend.iterrows():
        code = r["code"]
        if code in CH_CANTON_CENTROIDS:
            lat, lon = CH_CANTON_CENTROIDS[code]
            pretty = CODES[code].replace("st. ", "St. ").title()
            s = float(r["spend"])
            w_lat += lat * s
            w_lon += lon * s
            rows_map.append({"code": code, "lat": lat, "lon": lon, "spend": s, "name": pretty})
    map_df = pd.DataFrame(rows_map)

    if map_df.empty:
        st.info("Keine zuordenbaren Kantone gefunden – zeige Balkendiagramm.")
        agg = (regions.groupby("region_raw", as_index=False)["spend"].sum()
                      .sort_values("spend", ascending=False).head(20))
        st.plotly_chart(px.bar(agg, x="region_raw", y="spend", title="Top-Regionen (Budget-gewichtet)"),
                        use_container_width=True)
    else:
        center_lat = (w_lat / total_spend) if total_spend > 0 else 46.8
        center_lon = (w_lon / total_spend) if total_spend > 0 else 8.3

        # weniger dominante Kreise: kleinere Skala & geringere Opazität
        spend_max = float(map_df["spend"].max() or 0.0)
        def _bubble_size(s: float) -> float:
            base = 12.0
            scale = 40.0  # vorher 52.0 -> dezenter
            return base if spend_max <= 0 else base + scale * math.sqrt((s or 0.0) / spend_max)

        map_df["size"] = map_df["spend"].apply(_bubble_size)
        map_df["size_border"] = map_df["size"] + 4  # dezenter Ring

        # Hovertext
        hover_txt = [f"<b>{n}</b><br>Spend: {s:,.0f}".replace(",", " ")
                     for n, s in zip(map_df["name"], map_df["spend"])]

        fig_map = go.Figure()

        # Ring/Glow (weiß, sehr transparent) – Hintergrund bleibt gut sichtbar
        fig_map.add_trace(go.Scattermapbox(
            lat=map_df["lat"], lon=map_df["lon"],
            mode="markers",
            marker=dict(size=map_df["size_border"], color="rgba(255,255,255,1)"),
            hoverinfo="skip",
            opacity=0.30,  # vorher 0.85
            showlegend=False,
        ))

        # Füllkreis (blau, moderat transparent)
        fig_map.add_trace(go.Scattermapbox(
            lat=map_df["lat"], lon=map_df["lon"],
            mode="markers",
            marker=dict(size=map_df["size"], color="rgb(36,99,235)"),
            text=hover_txt, hoverinfo="text",
            opacity=0.55,  # vorher 0.88 -> weniger dominant
            showlegend=False,
        ))

        fig_map.update_layout(
            mapbox_style="carto-positron",  # Vorgabe beibehalten
            mapbox_center={"lat": center_lat, "lon": center_lon},
            mapbox_zoom=6.6,
            margin=dict(l=10, r=10, t=10, b=10),
            height=540,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Top-Regionen als Balken
    st.subheader("Top-Regionen (Spend)")
    top_codes = agg_spend.copy()
    top_codes["Kanton"] = top_codes["code"].map(lambda c: CODES.get(c, c).replace("st. ", "St. ").title())
    st.plotly_chart(px.bar(top_codes.sort_values("spend", ascending=False).head(20),
                           x="Kanton", y="spend", title=None, height=420),
                    use_container_width=True)
