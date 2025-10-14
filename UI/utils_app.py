# UI/utils_app.py
# Zentrale UI-Helfer für Navigation, Parsing, Aggregationen & Detailansicht

import os
import sys
import io
import json
from datetime import date, timedelta
from typing import Iterable, Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- (optional) Project-Root ins sys.path hängen, falls du außerhalb von UI importierst
_UI_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_UI_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# =========================
# Karten / Visuals
# =========================
def make_region_map(reg_sum: pd.DataFrame):
    """Erstellt eine einfache Bubble-Map (Bundesländer-Zentroiden) aus 'region' & 'spend'."""
    bl_centroids = {
        "Baden-Württemberg": (48.6616, 9.3501),
        "Bayern": (48.7904, 11.4979),
        "Berlin": (52.5200, 13.4050),
        "Brandenburg": (52.4125, 12.5316),
        "Bremen": (53.0793, 8.8017),
        "Hamburg": (53.5511, 9.9937),
        "Hessen": (50.6521, 9.1624),
        "Mecklenburg-Vorpommern": (53.6127, 12.4296),
        "Niedersachsen": (52.6367, 9.8451),
        "Nordrhein-Westfalen": (51.4332, 7.6616),
        "Rheinland-Pfalz": (49.9929, 7.8460),
        "Saarland": (49.3964, 6.9770),
        "Sachsen": (51.1045, 13.2017),
        "Sachsen-Anhalt": (51.9503, 11.6923),
        "Schleswig-Holstein": (54.2194, 9.6961),
        "Thüringen": (50.9010, 11.0375),
    }
    reg_sum = reg_sum.copy()
    reg_sum["region"] = reg_sum["region"].replace({"Saxony-Anhalt": "Sachsen-Anhalt"})

    rows_map = []
    for _, row in reg_sum.iterrows():
        name = row.get("region")
        if name in bl_centroids:
            lat, lon = bl_centroids[name]
            rows_map.append({"region": name, "lat": lat, "lon": lon, "spend": row.get("spend", 0)})

    if not rows_map:
        return px.scatter_mapbox(pd.DataFrame({"lat": [51], "lon": [10], "spend": [0]}),
                                 lat="lat", lon="lon", size="spend")

    map_df = pd.DataFrame(rows_map)
    fig = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lon", size="spend",
        hover_name="region", hover_data={"spend": ":.0f", "lat": False, "lon": False},
        zoom=5, height=520, title="Regionale Heatmap (Bubble-Intensity nach Spend)",
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, b=10, t=40))
    return fig


# =========================
# Navigation & Deep Links
# =========================
def _switch_page(target: str):
    try:
        st.switch_page(target)
    except Exception:
        # Fallback: QueryParam setzen (kann von Home ausgewertet werden)
        st.query_params["goto"] = target


def go_start(target_path: str):
    """Direkt auf eine Zielseite springen (z. B. 'pages/1_Upload.py')."""
    _switch_page(target_path)


def page_links(items: Iterable[Tuple[str, str]]):
    """Kleiner Link-Tree zu Subpages. items=[('Upload','pages/1_Upload.py'), ...]."""
    for label, path in items:
        if hasattr(st, "page_link"):
            st.page_link(path, label=label, icon=None)
        else:
            if st.button(label, key=f"nav_{label}"):
                _switch_page(path)


def nav_to(name_or_path: str):
    _switch_page(name_or_path)


def get_query_ad() -> Optional[str]:
    return st.query_params.get("ad")


def set_query_ad(ad_id: str):
    st.query_params["ad"] = str(ad_id)


# =========================
# DB-Connect (optional)
# =========================
@st.cache_resource(show_spinner=False)
def get_db():
    """Platzhalter für DB-Verbindung (psycopg2/SQLAlchemy)."""
    return None


# =========================
# Parsing / Import
# =========================
def _load_json_lines(f: io.BytesIO) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in f.getvalue().decode("utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out += [x for x in obj if isinstance(x, dict)]
        except Exception:
            continue
    return out


def _expand_meta_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    def safe_date(s):
        try:
            return pd.to_datetime(str(s)).date()
        except Exception:
            return None

    def midpoint(lo, hi):
        def f(x):
            try:
                return float(x)
            except Exception:
                return None

        lo, hi = f(lo), f(hi)
        if lo is None and hi is None:
            return None
        if lo is None:
            return hi
        if hi is None:
            return lo
        return (lo + hi) / 2.0

    text_pieces: List[str] = []
    for k in ["ad_creative_bodies", "ad_creative_link_titles", "ad_creative_link_descriptions"]:
        v = rec.get(k)
        if isinstance(v, list):
            text_pieces += [str(x) for x in v]
        elif isinstance(v, str):
            text_pieces.append(v)
    ad_text = "\n".join(text_pieces).strip()

    start = safe_date(rec.get("ad_delivery_start_time") or rec.get("ad_creation_time"))
    end = safe_date(rec.get("ad_delivery_stop_time")) or (start or date.today())
    if start and end and end < start:
        end = start

    impressions = midpoint(rec.get("impressions", {}).get("lower_bound"),
                           rec.get("impressions", {}).get("upper_bound"))
    spend = midpoint(rec.get("spend", {}).get("lower_bound"),
                     rec.get("spend", {}).get("upper_bound"))

    regions = rec.get("delivery_by_region") or rec.get("regions_json") or []
    age_country_gender = rec.get("age_country_gender_reach_breakdown") or []
    breakdown = age_country_gender[0]["age_gender_breakdowns"] if age_country_gender else []

    # creative_features robust einsammeln
    def _get_creative(rec_):
        if isinstance(rec_.get("creative_features"), dict):
            cf = rec_["creative_features"]
            return {
                "visuelle_features": cf.get("visuelle_features", {}) or {},
                "textuelle_features": cf.get("textuelle_features", {}) or {},
                "semantische_features": cf.get("semantische_features", {}) or {},
            }
        if isinstance(rec_.get("additional_data"), dict):
            add = rec_["additional_data"]
            return {
                "visuelle_features": add.get("visuelle_features", {}) or {},
                "textuelle_features": add.get("textuelle_features", {}) or {},
                "semantische_features": add.get("semantische_features", {}) or {},
            }
        return {
            "visuelle_features": rec_.get("visuelle_features", {}) or {},
            "textuelle_features": rec_.get("textuelle_features", {}) or {},
            "semantische_features": rec_.get("semantische_features", {}) or {},
        }

    cf = _get_creative(rec)
    if not any(cf.values()):
        cf = None

    return {
        "id": rec.get("id"),
        "advertiser": rec.get("page_name") or rec.get("bylines") or "Unbekannt",
        "campaign": (rec.get("ad_creative_link_titles") or ["Unbekannte Kampagne"])[0],
        "platforms": ", ".join(rec.get("publisher_platforms") or []),
        "ad_text": ad_text,
        "start_date": start,
        "end_date": end,
        "spend_mid": spend,
        "impressions_mid": impressions,
        "currency": rec.get("currency") or "EUR",
        "regions_count": len(regions),
        "regions_json": regions,
        "gender_age_json": rec.get("demographic_distribution") or [],
        "age_gender_breakdowns": breakdown,
        "snapshot_url": rec.get("ad_snapshot_url"),
        "creative_features": cf,
    }


@st.cache_data(show_spinner=False)
def parse_any_files(buffers: List[io.BytesIO], names: List[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for buf, name in zip(buffers, names):
        n = (name or "").lower()
        if n.endswith(".jsonl"):
            records += _load_json_lines(buf)
        elif n.endswith(".json"):
            obj = json.loads(buf.getvalue().decode("utf-8", errors="ignore"))
            records += obj if isinstance(obj, list) else [obj]
        elif n.endswith(".csv"):
            dfc = pd.read_csv(buf)
            if "raw" in dfc.columns:
                records += dfc["raw"].dropna().map(lambda x: json.loads(x) if isinstance(x, str) else x).tolist()
            else:
                records += dfc.to_dict(orient="records")
    if not records:
        return pd.DataFrame()
    rows = [_expand_meta_record(r) for r in records]
    df = pd.DataFrame(rows)
    # Fallback-Spalten
    if "creative" not in df.columns and "creative_features" in df.columns:
        df["creative"] = df["creative_features"]
    if "creative" not in df.columns:
        df["creative"] = [{} for _ in range(len(df))]
    # Heuristik (einfach)
    df["topics_heur"] = df["ad_text"].fillna("").map(lambda t: ["Sonstiges"] if not t else _topics_from_text(t))
    df["strategy_heur"] = df.apply(_strategy_heur, axis=1)
    return df


# =========================
# Heuristiken
# =========================
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Klima & Energie": ["klima", "klimaschutz", "co2", "energie", "solar", "photovoltaik", "wind", "erneuerbar"],
    "Migration & Integration": ["migration", "migrant", "flücht", "asyl", "integration", "grenze", "abschieb"],
    "Soziales & Verteilung": ["sozial", "rente", "löhne", "miete", "lebenshaltung", "armut", "gerecht", "bürgergeld"],
    "Mobilität & ÖV": ["öpnv", "öpv", "ticket", "bahn", "takt", "pendler"],
    "Sicherheit & Ordnung": ["polizei", "kriminal", "innere sicherheit", "gewalt", "terror"],
    "Außen & Entwicklung": ["afrika", "eu", "ukraine", "nato", "entwicklungshilfe", "ausland"],
    "Wirtschaft & Innovation": ["wirtschaft", "innovation", "arbeitsplätze", "jobs", "industrie", "mittelstand"],
}


def _topics_from_text(text: str) -> List[str]:
    t = (text or "").lower()
    hits = [k for k, kws in TOPIC_KEYWORDS.items() if any(kw in t for kw in kws)]
    return hits[:3] if hits else ["Sonstiges"]


def _strategy_heur(row: pd.Series) -> List[str]:
    strategy: List[str] = []
    audience_div = len(row.get("age_gender_breakdowns") or [])
    regions_n = int(row.get("regions_count") or 0)
    txt = f"{row.get('ad_text', '')}".lower()
    if audience_div >= 5 and regions_n >= 8:
        strategy.append("Hypertargeting")
    if any(w in txt for w in ["jetzt", "unterstützen", "wählen", "stimmen", "teilen"]):
        strategy.append("Mobilisierung")
    if any(w in txt for w in ["fair", "kosten", "arbeit", "jobs", "innovation", "leistung"]):
        strategy.append("Persuasion")
    return strategy or ["Unklar"]


# =========================
# Aggregationen (gecached)
# =========================
@st.cache_data(show_spinner=False)
def compute_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sd, ed = r.get("start_date"), r.get("end_date")
        if pd.isna(sd) or pd.isna(ed):
            continue
        sd, ed = pd.to_datetime(sd).date(), pd.to_datetime(ed).date()
        days = max(1, (ed - sd).days + 1)
        spend_per = float(r.get("spend_mid") or 0) / days
        imp_per = float(r.get("impressions_mid") or 0) / days
        for i in range(days):
            d = sd + timedelta(days=i)
            rows.append({
                "date": pd.to_datetime(d),
                "id": r.get("id"),
                "campaign": r.get("campaign"),
                "spend": spend_per,
                "impressions": imp_per,
                "topics": r.get("topics") or r.get("topics_heur") or ["Sonstiges"],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_topics_ts(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sd, ed = r.get("start_date"), r.get("end_date")
        if pd.isna(sd) or pd.isna(ed):
            continue
        sd, ed = pd.to_datetime(sd).date(), pd.to_datetime(ed).date()
        days = max(1, (ed - sd).days + 1)
        per_day_spend = float(r.get("spend_mid") or 0) / days
        topics = r.get("topics") or r.get("topics_heur") or ["Sonstiges"]
        for i in range(days):
            dt = sd + timedelta(days=i)
            for t in topics:
                rows.append({"date": pd.to_datetime(dt), "topic": t, "spend": per_day_spend / len(topics)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_audience_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        dem = r.get("gender_age_json") or []
        s = float(r.get("spend_mid") or 0.0)
        for dmp in dem:
            try:
                p = float(dmp.get("percentage"))
            except Exception:
                p = None
            rows.append({"audience": f"{dmp.get('age','?')} · {dmp.get('gender','?')}", "spend": s * (p if p is not None else 0)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_regions_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        regs = r.get("delivery_by_region") or r.get("regions_json") or []
        s = float(r.get("spend_mid") or 0.0)
        for rg in regs:
            try:
                p = float(rg.get("percentage"))
            except Exception:
                p = None
            rows.append({"region": rg.get("region", "?"), "spend": s * (p if p is not None else 0)})
    return pd.DataFrame(rows)


# =========================
# Alerts
# =========================
def _kl_div(P, Q, eps=1e-9):
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    return float(np.sum(P * np.log(P / Q)))


@st.cache_data(show_spinner=False)
def find_shifts_spikes(
    daily: pd.DataFrame,
    topics_ts: pd.DataFrame,
    N: int = 5,
    kl_thr: float = 0.05,
    last_w: int = 4,
    prev_w: int = 4,
    spike_ratio: float = 1.4,
):
    alerts: List[Tuple[str, str]] = []
    shift_table = pd.DataFrame()
    drivers = pd.DataFrame()

    # Themen-Shift
    if not topics_ts.empty and topics_ts["date"].nunique() >= 2 * N:
        agg = topics_ts.groupby(["date", "topic"], as_index=False)["spend"].sum()
        dates_sorted = sorted(agg["date"].unique())
        last_dates, prev_dates = dates_sorted[-N:], dates_sorted[-2 * N : -N]
        p = agg[agg["date"].isin(last_dates)].groupby("topic")["spend"].sum()
        q = agg[agg["date"].isin(prev_dates)].groupby("topic")["spend"].sum()
        keys = sorted(set(p.index).union(q.index))
        P = np.array([p.get(k, 0.0) for k in keys])
        Q = np.array([q.get(k, 0.0) for k in keys])
        if P.sum() > 0 and Q.sum() > 0:
            P, Q = P / P.sum(), Q / Q.sum()
            kld = _kl_div(P, Q)
            delta = pd.Series(P - Q, index=keys).sort_values(key=np.abs, ascending=False)
            if kld > kl_thr:
                msg = ", ".join([f"{k} ({'+' if v>0 else ''}{v*100:.1f}pp)" for k, v in delta.head(3).items()])
                alerts.append(("🧭 Strategischer Shift", f"{msg} (KL={kld:.3f})"))
                shift_table = (
                    pd.DataFrame(
                        {"Topic": keys, "Anteil vorher (%)": Q * 100, "Anteil zuletzt (%)": P * 100, "Δ Prozentpunkte": (P - Q) * 100}
                    )
                    .sort_values("Δ Prozentpunkte", key=lambda s: s.abs(), ascending=False)
                )

    # Spend-Spikes + Treiber
    if not daily.empty and len(daily["date"].unique()) >= (last_w + prev_w):
        spend_by_day = daily.groupby("date", as_index=False)["spend"].sum().sort_values("date")
        last_mean = spend_by_day.tail(last_w)["spend"].mean()
        prev_mean = spend_by_day.tail(last_w + prev_w).head(prev_w)["spend"].mean()
        ratio = (last_mean / prev_mean) if prev_mean > 0 else np.inf
        if prev_mean > 0 and ratio >= spike_ratio:
            alerts.append(("📈 Kommunikationsdruck-Spike", f"Ø Spend letzte {last_w} Tage {last_mean:,.0f} vs. davor {prev_mean:,.0f} (x{ratio:.2f})".replace(",", ".")))

            last_days = spend_by_day["date"].tail(last_w).tolist()
            prev_days = spend_by_day["date"].tail(last_w + prev_w).head(prev_w).tolist()

            per_ad_day = daily.groupby(["id", "campaign", "date"], as_index=False)["spend"].sum()
            by_prev = per_ad_day[per_ad_day["date"].isin(prev_days)].groupby(["id", "campaign"], as_index=False)["spend"].sum().rename(columns={"spend": "spend_prev"})
            by_last = per_ad_day[per_ad_day["date"].isin(last_days)].groupby(["id", "campaign"], as_index=False)["spend"].sum().rename(columns={"spend": "spend_last"})
            dv = pd.merge(by_last, by_prev, on=["id", "campaign"], how="outer").fillna(0.0)
            dv["Δ Spend"] = dv["spend_last"] - dv["spend_prev"]
            drivers = dv.sort_values("Δ Spend", ascending=False)

    return alerts, shift_table, drivers


# =========================
# Tabellen (AgGrid optional)
# =========================
def aggrid_or_dataframe(df: pd.DataFrame, height: int = 420, key: str = "grid"):
    """Nutzt AgGrid falls installiert, sonst Fallback auf st.dataframe."""
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode  # type: ignore
        gob = GridOptionsBuilder.from_dataframe(df)
        gob.configure_grid_options(rowHeight=28)
        gob.configure_default_column(resizable=True, sortable=True, filter=True)
        grid = AgGrid(
            df,
            gridOptions=gob.build(),
            height=height,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.NO_UPDATE,
            key=key,
        )
        return grid
    except Exception:
        st.dataframe(df, height=height, use_container_width=True)
        return None


# =========================
# Detail-View (Ad)
# =========================
def ad_detail_view(ad_id: str):
    df = st.session_state.get("raw_import_df") or pd.DataFrame()
    row = df[df["id"].astype(str) == str(ad_id)]
    if row.empty:
        st.warning("Ad nicht gefunden.")
        return
    r = row.iloc[0]
    tabs = st.tabs(["Overview", "Text & Themen", "Strategie", "Schwachstellen", "Media", "Trace"])
    with tabs[0]:
        st.markdown(f"**Akteur:** {r.get('advertiser','-')}  \n**Kampagne:** {r.get('campaign','-')}")
        st.markdown(f"**Zeitraum:** {r.get('start_date')} → {r.get('end_date')}  \n**Plattformen:** {r.get('platforms')}")
        st.markdown(f"**Spend (mid):** {r.get('spend_mid')}  ·  **Impressions (mid):** {r.get('impressions_mid')}")
    with tabs[1]:
        st.text_area("Ad-Text", value=r.get("ad_text") or "", height=160)
        topics = r.get("topics") or r.get("topics_heur") or []
        st.write("**Themen:**", ", ".join(topics))
    with tabs[2]:
        strat = r.get("strategy") or r.get("strategy_heur") or []
        st.write("**Strategie:**", ", ".join(strat))
    with tabs[3]:
        weak = st.session_state.get("weaknesses_results")
        if isinstance(weak, pd.DataFrame):
            wrow = weak[weak["id"].astype(str) == str(ad_id)]
            st.dataframe(wrow, use_container_width=True) if not wrow.empty else st.caption("Keine Schwachstellen-Daten.")
        else:
            st.caption("Keine Schwachstellen-Daten.")
    with tabs[4]:
        url = r.get("snapshot_url")
        st.link_button("Ad Snapshot öffnen", url) if url else st.caption("Kein Snapshot-Link vorhanden.")
    with tabs[5]:
        traces = (
            st.session_state.get("ad_tagging_params", {}).get(str(ad_id)),
            st.session_state.get("ad_tagging_usage", {}).get(str(ad_id)),
            st.session_state.get("ad_tagging_raw", {}).get(str(ad_id)),
        )
        st.json({"params": traces[0], "usage": traces[1], "raw": traces[2]}) if any(traces) else st.caption("Keine Trace-Daten.")
