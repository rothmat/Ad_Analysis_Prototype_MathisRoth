# pages/08_Schwachstellen_Risiken.py
# -*- coding: utf-8 -*-
import os, json, math, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# --- OpenAI-gestützte Schwachstellen-Analyse (dein Script)
from agents.weaknesses_analyzer import analyze_weaknesses

# -------------------- DB client --------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

# -------------------- Page --------------------
st.set_page_config(page_title="Schwachstellen & Risiken", page_icon="🛡️", layout="wide")
st.title("🛡️ Schwachstellen & Risiken")

from time import time
PROGRESS_PATH = Path(__file__).resolve().parents[2] / ".run" / "weak_progress.json"
PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _write_progress(**kw):
    try:
        PROGRESS_PATH.write_text(json.dumps({
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            **kw
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# -------------------- Scoped CSS nur für ADS-Auswahl --------------------
st.markdown("""
<style>
/* Labels NUR im Ads-Select verbergen, Filter-Labels bleiben sichtbar */
#ads-select-scope div[data-testid="stMultiSelect"] label { display:none; }
/* Button rechts minimal nach oben ziehen, damit er auf gleicher Höhe ist */
#ads-select-scope div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
  margin-top: -2px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DB helpers --------------------

def _safe_rb(conn):
    try:
        conn.rollback()
    except Exception:
        pass

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
    out=[]
    for ad_pk, snapshot_date, fused, _ in rows:
        if isinstance(fused, str):
            try: fused = json.loads(fused)
            except Exception: fused = {}
        out.append({"ad_pk": int(ad_pk), "snapshot_date": str(snapshot_date), "fused": fused})
    return pd.DataFrame(out)

# -------------------- Extractors --------------------
def _media_id_from_raw(raw: Dict[str,Any]) -> Optional[str]:
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

def _get_creative(api_raw: Dict[str,Any]) -> Dict[str,Any]:
    if isinstance(api_raw.get("creative_features"), dict):
        cf = api_raw["creative_features"]
        return {
            "visuelle_features": cf.get("visuelle_features", {}) or {},
            "textuelle_features": cf.get("textuelle_features", {}) or {},
            "semantische_features": cf.get("semantische_features", {}) or {},
        }
    if isinstance(api_raw.get("additional_data"), dict):
        add = api_raw["additional_data"]
        return {
            "visuelle_features": add.get("visuelle_features", {}) or {},
            "textuelle_features": add.get("textuelle_features", {}) or {},
            "semantische_features": add.get("semantische_features", {}) or {},
        }
    return {
        "visuelle_features": api_raw.get("visuelle_features", {}) or {},
        "textuelle_features": api_raw.get("textuelle_features", {}) or {},
        "semantische_features": api_raw.get("semantische_features", {}) or {},
    }

# -------------------- DB ensure / save --------------------
def ensure_weakness_table(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ad_weaknesses (
                    id SERIAL PRIMARY KEY,
                    ad_id INTEGER NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("ALTER TABLE ad_weaknesses ADD COLUMN IF NOT EXISTS model TEXT;")
            cur.execute("ALTER TABLE ad_weaknesses ADD COLUMN IF NOT EXISTS result_json JSONB;")
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE schemaname = ANY (current_schemas(true))
                          AND indexname = 'ad_weaknesses_ad_model_uniq'
                    ) THEN
                        CREATE UNIQUE INDEX ad_weaknesses_ad_model_uniq
                        ON ad_weaknesses(ad_id, model);
                    END IF;
                END$$;
            """)
        conn.commit()
    except Exception:
        _safe_rb(conn)
        raise

def _table_columns(conn, table="ad_weaknesses") -> set[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name=%s AND table_schema = ANY (current_schemas(true));
        """, (table,))
        return {r[0] for r in cur.fetchall()}

def _has_unique_ad_model(conn) -> bool:
    # Prüft, ob es einen Unique-Index über (ad_id, model) gibt
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = ANY (current_schemas(true))
              AND indexname = 'ad_weaknesses_ad_model_uniq'
        """)
        return cur.fetchone() is not None

def _has_pkey_on_ad_id(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = 'ad_weaknesses'::regclass
              AND i.indisprimary
        """)
        cols = [r[0] for r in cur.fetchall()]
    return cols == ["ad_id"] or cols == ["ad_id",]  # robust

def save_weaknesses(conn, df_out: pd.DataFrame, model_name: str) -> int:
    if df_out is None or df_out.empty:
        return 0
    ensure_weakness_table(conn)
    cols = _table_columns(conn, "ad_weaknesses")

    json_col = "result_json" if "result_json" in cols else ("analysis" if "analysis" in cols else None)
    if json_col is None:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE ad_weaknesses ADD COLUMN IF NOT EXISTS result_json JSONB;")
        conn.commit()
        json_col = "result_json"

    has_risk  = "overall_risk" in cols
    has_conf  = "overall_confidence" in cols
    has_upd   = "updated_at" in cols

    # Konflikt-Strategie abhängig vom Schema
    use_pair_unique = _has_unique_ad_model(conn)
    use_pkey_ad_id  = _has_pkey_on_ad_id(conn)

    n = 0
    with conn.cursor() as cur:
        for row in df_out.itertuples():
            try:
                ad_id = int(getattr(row, "id"))
            except Exception:
                continue

            payload = df_out.loc[row.Index].to_dict()
            values = [ad_id, model_name, json.dumps(payload, ensure_ascii=False)]
            col_names = ["ad_id", "model", json_col]

            if has_risk:
                col_names.append("overall_risk");        values.append(float(payload.get("overall_risk") or 0))
            if has_conf:
                col_names.append("overall_confidence");  values.append(float(payload.get("overall_confidence") or 0))

            placeholders = ", ".join(["%s"] * len(col_names))
            set_parts = [f"{c}=EXCLUDED.{c}" for c in col_names if c not in ("ad_id","model")]
            if has_upd:
                set_parts.append("updated_at=NOW()")

            if use_pair_unique:
                conflict = "ON CONFLICT (ad_id, model)"
            elif use_pkey_ad_id:
                conflict = "ON CONFLICT ON CONSTRAINT ad_weaknesses_pkey"
            else:
                # Fallback: Konflikt nur auf ad_id (wenn PK anderweitig)
                conflict = "ON CONFLICT (ad_id)"

            sql = f"""
                INSERT INTO ad_weaknesses ({", ".join(col_names)})
                VALUES ({placeholders})
                {conflict} DO UPDATE SET {", ".join(set_parts)};
            """
            cur.execute(sql, values)
            n += 1

    conn.commit()
    return n

# -------------------- Connect --------------------
try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")
    st.stop()

# -------------------- Kampagne wählen --------------------
camp_opts = get_campaign_options(conn)
if not camp_opts:
    st.info("Keine Kampagnen gefunden."); st.stop()

labels = [f"{n} ({s})" for _, n, s in camp_opts]
idx = st.selectbox("Kampagne", options=list(range(len(camp_opts))), format_func=lambda i: labels[i])
campaign_id, campaign_name, campaign_slug = camp_opts[idx]

# -------------------- Ads + Labels bauen --------------------
ads_df = get_llm_ads_for_campaign(conn, campaign_slug)
all_ad_pks = [int(r.ad_pk) for r in ads_df.itertuples()]
_meta_rows = load_fused_rows(conn, campaign_slug, all_ad_pks)

latest_meta: Dict[int, Dict[str,str | None]] = {}
if not _meta_rows.empty:
    for ad_pk, grp in _meta_rows.groupby("ad_pk", sort=False):
        last = grp.tail(1).iloc[0]
        api_raw = (last["fused"].get("api") or {}).get("raw") or {}
        latest_meta[int(ad_pk)] = {
            "page_name": (_extract_group(api_raw) or None),
            "media_id":  (_media_id_from_raw(api_raw) or None),
        }

def _label_for_ad(ad_pk: int, ad_external_id: Any) -> str:
    meta = latest_meta.get(int(ad_pk), {})
    pn = meta.get("page_name") or "-"
    mid = meta.get("media_id") or (str(ad_external_id) if ad_external_id is not None else "-")
    return f"{pn} ({mid})"

ad_labels = [_label_for_ad(int(r.ad_pk), r.ad_external_id) for r in ads_df.itertuples()]
label_to_pk = {_label_for_ad(int(r.ad_pk), r.ad_external_id): int(r.ad_pk) for r in ads_df.itertuples()}

# -------------------- Auswahl + Button (auf gleicher Höhe) --------------------
st.subheader("Ads auswählen")
st.markdown('<div id="ads-select-scope">', unsafe_allow_html=True)

SEL_KEY  = "sel_ads_weak"
ALL_FLAG = "__sel_ads_weak_all_flag__"

# 1) „Alle auswählen“ Flag VOR dem Widget auswerten
if st.session_state.get(ALL_FLAG):
    st.session_state[SEL_KEY] = ad_labels[:]
    st.session_state.pop(ALL_FLAG, None)

# 2) Widget rendern
col_sel, col_btn = st.columns([6, 1], vertical_alignment="center")
with col_sel:
    prev = st.session_state.get(SEL_KEY, ad_labels[:min(8, len(ad_labels))])
    default_vals = [v for v in prev if v in ad_labels] or ad_labels[:min(8, len(ad_labels))]
    sel_labels = st.multiselect(
        "Ausgewählte Ads",
        options=ad_labels,
        default=default_vals,
        key=SEL_KEY,
    )
with col_btn:
    if st.button("Alle auswählen", use_container_width=True, key="btn_select_all"):
        st.session_state[ALL_FLAG] = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

selected_ad_pks = [label_to_pk[l] for l in st.session_state.get(SEL_KEY, []) if l in label_to_pk]
st.divider()

# -------------------- Optionale Filter (Gruppe/Sponsoren) --------------------
st.subheader("Filter (optional)")
c0, c1, c2 = st.columns([1,3,3])
with c0:
    enable_group = st.checkbox("Gruppe filtern", value=False, key="fltr_group")
    enable_spon  = st.checkbox("Sponsoren filtern", value=False, key="fltr_spon")

fused_df = load_fused_rows(conn, campaign_slug, selected_ad_pks)
if fused_df.empty:
    st.info("Keine Daten zu den ausgewählten Ads."); st.stop()

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
        key="sel_groups"
    )
with c2:
    sel_sponsors = st.multiselect(
        "Sponsoren (bylines)",
        options=all_sponsors if all_sponsors else [],
        default=all_sponsors if all_sponsors else [],
        disabled=not enable_spon,
        key="sel_sponsors"
    )

mask = pd.Series(True, index=fused_df.index)
if enable_group and sel_groups:
    mask &= fused_df["__group"].isin(set(sel_groups))
if enable_spon and sel_sponsors:
    sset = set(sel_sponsors)
    mask &= fused_df["__sponsors"].apply(lambda lst: any(s in sset for s in lst))

filtered = fused_df[mask].copy()
if filtered.empty:
    st.info("Keine Daten nach den Filtern. Bereits vorhandene Ergebnisse werden unten angezeigt (falls vorhanden).")

# -------------------- Helpers: Tabellen-Checks & Counts --------------------
def _table_exists(conn, table_name: str) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
            return bool(cur.fetchone()[0])
    except Exception:
        _safe_rb(conn)
        return False

def _table_columns(conn, table="ad_weaknesses") -> set[str]:
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name=%s AND table_schema = ANY (current_schemas(true));
            """, (table,))
            return {r[0] for r in cur.fetchall()}
    except Exception:
        _safe_rb(conn)
        return set()

def _count_weaknesses(conn, ad_ids: List[int], model: Optional[str] = None) -> int:
    if not ad_ids or not _table_exists(conn, "ad_weaknesses"):
        return 0
    try:
        cols = _table_columns(conn, "ad_weaknesses")
        with conn.cursor() as cur:
            if model and "model" in cols:
                cur.execute("""
                    SELECT COUNT(DISTINCT ad_id)
                    FROM ad_weaknesses
                    WHERE ad_id = ANY(%s) AND model = %s
                """, (ad_ids, model))
            else:
                cur.execute("""
                    SELECT COUNT(DISTINCT ad_id)
                    FROM ad_weaknesses
                    WHERE ad_id = ANY(%s)
                """, (ad_ids,))
            n = cur.fetchone()[0] or 0
        return int(n)
    except Exception:
        _safe_rb(conn)
        return 0

def _existing_weakness_ids(conn, ad_ids: list[int], model: str) -> set[int]:
    if not ad_ids or not _table_exists(conn, "ad_weaknesses"):
        return set()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ad_id
                FROM ad_weaknesses
                WHERE ad_id = ANY(%s) AND model = %s
            """, (ad_ids, model))
            return {int(r[0]) for r in cur.fetchall()}
    except Exception:
        _safe_rb(conn)
        return set()
    
def _save_df(df_out: pd.DataFrame):
    try:
        n = save_weaknesses(conn, df_out, model_name=model)
        return n, None
    except Exception as e:
        _safe_rb(conn)  # <<< wichtig: Transaktion zurücksetzen
        return 0, str(e)
    
def fetch_weaknesses_from_db(
    conn,
    campaign_slug: str,
    model: Optional[str] = None,
    ad_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Holt alle Schwachstellen-Ergebnisse aus ad_weaknesses für eine Kampagne.
    Erkennt automatisch, ob 'result_json' existiert (neues Schema) oder
    nur Spalten wie overall_risk / overall_confidence (altes Schema).
    """
    # Spalten prüfen
    cols = _table_columns(conn, "ad_weaknesses")
    has_json = "result_json" in cols
    has_risk = "overall_risk" in cols
    has_conf = "overall_confidence" in cols

    where = ["c.slug = %s"]
    params: List[Any] = [campaign_slug]

    if model:
        where.append("w.model = %s")
        params.append(model)

    if ad_ids:
        where.append("w.ad_id = ANY(%s)")
        params.append(ad_ids)

    where_sql = " AND ".join(where)

    sel_cols = [
        "w.ad_id",
        "w.model",
        "w.updated_at",
    ]
    if has_json:
        sel_cols.append("w.result_json")
    if has_risk:
        sel_cols.append("w.overall_risk")
    if has_conf:
        sel_cols.append("w.overall_confidence")

    sql = f"""
        SELECT {", ".join(sel_cols)}
        FROM ad_weaknesses w
        JOIN ads a        ON a.id = w.ad_id
        JOIN campaigns c  ON c.id = a.campaign_id
        WHERE {where_sql}
        ORDER BY w.updated_at DESC, w.ad_id DESC
    """

    rows = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for rec in cur.fetchall():
            # Mapping je nach ausgewählten Spalten
            i = 0
            ad_id       = int(rec[i]); i += 1
            mdl         = rec[i];      i += 1
            updated_at  = rec[i];      i += 1
            result_json = rec[i] if has_json else None
            if has_json: i += 1
            overall_risk        = float(rec[i]) if has_risk else None
            if has_risk: i += 1
            overall_confidence  = float(rec[i]) if has_conf else None

            base = {
                "id": ad_id,
                "model": mdl,
                "updated_at": updated_at,
            }

            if result_json:
                try:
                    payload = result_json if isinstance(result_json, dict) else json.loads(result_json)
                except Exception:
                    payload = {}
                # sicherstellen, dass id drin ist
                payload = dict(payload or {})
                payload.setdefault("id", ad_id)
                payload.setdefault("model", mdl)
                payload.setdefault("updated_at", str(updated_at))
                # Fallback: Werte aus Spalten übernehmen, falls im JSON fehlen
                if overall_risk is not None:
                    payload.setdefault("overall_risk", overall_risk)
                if overall_confidence is not None:
                    payload.setdefault("overall_confidence", overall_confidence)
                rows.append(payload)
            else:
                # altes Schema ohne JSON
                base["overall_risk"] = overall_risk if overall_risk is not None else 0.0
                base["overall_confidence"] = overall_confidence if overall_confidence is not None else 0.0
                rows.append(base)

    return pd.DataFrame(rows)

# -------------------- Analyse starten (Batch-Runner) --------------------
st.subheader("Analyse starten")

api_key = os.getenv("OPENAI_API_KEY") or ""
model   = st.selectbox("Modell", ["gpt-4o-mini","gpt-4o","gpt-4.1"], index=0, key="weak_model")

# Status-Kacheln (wie gehabt)
n_sel = len(selected_ad_pks)
already_any   = _count_weaknesses(conn, selected_ad_pks, model=None)
already_model = _count_weaknesses(conn, selected_ad_pks, model=model)

cA, cB, cC = st.columns([1,1,2])
cA.metric("Ausgewählt", n_sel)
cB.metric(f"Schon analysiert ({model})", already_model)
with cC:
    st.caption(f"↳ Insgesamt bereits analysiert (irgendein Modell): **{already_any}/{n_sel}**")

# Force-Toggle: bereits analysierte (gleiches Modell) erneut rechnen?
force = st.toggle("Erneut analysieren", value=False, key="weak_force")

already_model_ids = _existing_weakness_ids(conn, selected_ad_pks, model)
ids_to_run = selected_ad_pks if force else [i for i in selected_ad_pks if i not in already_model_ids]

st.caption(
    f"Ausgewählt: **{len(selected_ad_pks)}** · bereits analysiert ({model}): **{len(already_model_ids)}** · "
    f"werden jetzt analysiert: **{len(ids_to_run)}**"
)

# Fortschrittsbalken (DB-basiert)
prog_box = st.empty()
def _render_progress():
    total = max(1, len(selected_ad_pks))
    done  = _count_weaknesses(conn, selected_ad_pks, model=model)
    prog_box.progress(min(1.0, done/total), text=f"Analysiert ({model}): {done} / {total}")

_render_progress()

# ---------- Batch-Queue im Session State ----------
Q_KEY   = "weak_queue"        # ad_id Rest-Queue
RUN_KEY = "weak_running"      # Flag laufender Job
ERR_KEY = "weak_errors"       # gesammelte Fehlerstrings

if ERR_KEY not in st.session_state:
    st.session_state[ERR_KEY] = []

# Buttons: Start / Stop
col_start, col_stop = st.columns([1,1])
with col_start:
    if st.button("🛡️ Analyse starten (Batch)", type="primary", key="weak_btn_start"):
        st.session_state[Q_KEY]   = ids_to_run[:]  # Snapshot der aktuellen Auswahl
        st.session_state[RUN_KEY] = True
        st.session_state[ERR_KEY] = []
        _write_progress(status="running", queued=len(st.session_state[Q_KEY]), model=model)
        st.rerun()

with col_stop:
    if st.button("⏹️ Abbrechen", key="weak_btn_stop", disabled=not st.session_state.get(RUN_KEY)):
        st.session_state[RUN_KEY] = False
        st.session_state.pop(Q_KEY, None)
        _write_progress(status="stopped", queued=0, model=model)
        st.experimental_set_query_params(ts=str(time()))  # entkoppelt Cache
        st.rerun()

# ---------- Worker-Tick: verarbeitet in jedem Run N Ads ----------
BATCH_SIZE = 5

def _to_rows_for_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for row in df.itertuples():
        api = (row.fused.get("api") or {}).get("raw") or {}
        text_pieces=[]
        for k in ["ad_creative_bodies","ad_creative_link_titles","ad_creative_link_descriptions"]:
            v = api.get(k)
            if isinstance(v, list): text_pieces.extend([str(x) for x in v])
            elif isinstance(v, str): text_pieces.append(v)
        ad_text="\n".join(text_pieces).strip()
        start = api.get("ad_delivery_start_time") or api.get("ad_creation_time")
        end   = api.get("ad_delivery_stop_time") or start
        plats = ", ".join(api.get("publisher_platforms") or [])
        rows.append({
            "id": int(row.ad_pk),
            "ad_text": ad_text,
            "platforms": plats,
            "start_date": start,
            "end_date": end,
            "creative_features": _get_creative(api),
        })
    return pd.DataFrame(rows)

def _save_df(df_out: pd.DataFrame):
    try:
        n = save_weaknesses(conn, df_out, model_name=model)
        return n, None
    except Exception as e:
        return 0, str(e)

# Wenn ein Lauf aktiv ist → in Batches abarbeiten und UI zyklisch rerendern
if st.session_state.get(RUN_KEY) and st.session_state.get(Q_KEY):
    queue: List[int] = st.session_state[Q_KEY]
    batch_ids = queue[:BATCH_SIZE]

    # Subset aus fused_df nur für diese IDs
    this_df = fused_df[fused_df["ad_pk"].isin(batch_ids)]
    subset  = _to_rows_for_analyzer(this_df)

    with st.spinner(f"Analysiere {len(batch_ids)} Ads …"):
        try:
            df_out = analyze_weaknesses(subset, api_key=api_key, model=model)
        except Exception as e:
            st.session_state[ERR_KEY].append(f"Analyze-Fehler: {e}")
            df_out = pd.DataFrame()

    # Session-Ergebnis aktualisieren (anhängen/vereinigen)
    prev = st.session_state.get("weak_last")
    if isinstance(prev, pd.DataFrame) and not prev.empty:
        try:
            # gleiche Struktur → lieber zusammenführen und nach id de-duplizieren
            comb = pd.concat([prev, df_out], ignore_index=True)
            if "id" in comb.columns:
                comb = comb.drop_duplicates(subset=["id"], keep="last")
            st.session_state["weak_last"] = comb
        except Exception:
            st.session_state["weak_last"] = df_out
    else:
        st.session_state["weak_last"] = df_out

    saved = 0
    if isinstance(df_out, pd.DataFrame) and not df_out.empty:
        s, err = _save_df(df_out)
        saved += s
        if err: st.session_state[ERR_KEY].append(f"Save-Fehler: {err}")

    # Queue kürzen, Fortschritt schreiben
    st.session_state[Q_KEY] = queue[BATCH_SIZE:]
    left = len(st.session_state[Q_KEY])
    _write_progress(status="running", queued=left, saved=saved, last_batch=batch_ids, model=model)

    # Fortschritt neu zeichnen
    _render_progress()

    # Nächster Tick (Auto-Refresh)
    import time as _t
    _t.sleep(0.5)
    st.rerun()

# Lauf fertig?
if st.session_state.get(RUN_KEY) and not st.session_state.get(Q_KEY):
    st.session_state[RUN_KEY] = False
    _write_progress(status="done", queued=0, model=model)
    st.success("Analyse abgeschlossen.")
    _render_progress()

# Fehler-Liste (falls etwas im Batch schiefging)
errs = st.session_state.get(ERR_KEY) or []
if errs:
    with st.expander("Fehlermeldungen (letzter Lauf)", expanded=False):
        for e in errs[-50:]:
            st.error(e)

def fetch_weaknesses_by_ids(conn, ad_ids: list[int], model: str | None = None) -> pd.DataFrame:
    if not ad_ids or not _table_exists(conn, "ad_weaknesses"):
        return pd.DataFrame()
    cols = _table_columns(conn, "ad_weaknesses")
    use_json = "result_json" in cols
    sql = """
        SELECT ad_id, model, {col}, overall_risk, overall_confidence
        FROM ad_weaknesses
        WHERE ad_id = ANY(%s) {model_clause}
    """.format(
        col = "result_json" if use_json else "analysis",
        model_clause = "AND model = %s" if (model and "model" in cols) else ""
    )
    with conn.cursor() as cur:
        cur.execute(sql, (ad_ids, model) if (model and "model" in cols) else (ad_ids,))
        rows = cur.fetchall()

    recs = []
    for ad_id, mdl, payload, risk, conf in rows:
        try:
            rec = payload if isinstance(payload, dict) else json.loads(payload)
            if not isinstance(rec, dict):
                continue
        except Exception:
            continue
        # Felder sicherstellen
        rec = {**rec}
        rec["id"] = int(rec.get("id") or ad_id)
        rec.setdefault("overall_risk",       float(risk) if risk is not None else 0.0)
        rec.setdefault("overall_confidence", float(conf) if conf is not None else 0.0)
        recs.append(rec)

    return pd.DataFrame(recs)

# -------------------Ergebnisse anzeigen -------------------------
st.subheader("Ergebnisse anzeigen")

source_mode = st.radio(
    "Quelle",
    ["Alle aus DB (empfohlen)", "Nur diese Sitzung"],
    index=0,
    horizontal=True,
)

if source_mode.startswith("Alle"):
    only_this_model = st.toggle("Nur dieses Modell filtern", value=False, key="weak_only_model")
    model_filter = model if only_this_model else None

    df_out = fetch_weaknesses_from_db(
        conn=conn,
        campaign_slug=campaign_slug,   # <— jetzt Teil der Signatur
        model=model_filter,
        ad_ids=None,                   # oder: selected_ad_pks, wenn du nur Auswahl sehen willst
    )
else:
    df_out = st.session_state.get("weak_last")
    if not isinstance(df_out, pd.DataFrame):
        df_out = pd.DataFrame()

if df_out.empty:
    st.caption("Keine Ergebnisse gefunden. Führe eine Analyse aus oder wechsle auf „Alle aus DB“.")
    st.stop()

# ---- Meta für Ergebnis-Filter aus fused_df holen ----
# ---- Meta-Quelle bestimmen (Alle aus DB => alle Ads der Kampagne) ----
if source_mode.startswith("Alle"):
    meta_source_df = load_fused_rows(conn, campaign_slug, all_ad_pks)  # alle Ads dieser Kampagne
else:
    meta_source_df = fused_df  # nur selektierte

meta_rows = []
for r in meta_source_df.itertuples():
    api_raw = (r.fused.get("api") or {}).get("raw") or {}
    meta_rows.append({
        "id": str(int(r.ad_pk)),
        "group": _extract_group(api_raw),
        "sponsors_list": _extract_sponsors(api_raw),
    })

meta_df = pd.DataFrame(meta_rows)
meta_exploded = meta_df.explode("sponsors_list") if not meta_df.empty else pd.DataFrame(columns=["id","group","sponsors_list"])

# ---- Ergebnisse mit Meta mergen (für Filter) ----
res = df_out.copy()
res["id"] = res["id"].astype(str)
res = res.merge(meta_exploded, on="id", how="left")
res.rename(columns={"sponsors_list": "sponsor"}, inplace=True)

# -------------------- Filter (Ergebnisse) --------------------
st.subheader("Filter (Ergebnisse)")

# NEU: vertikal ausrichten
fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")

with fcol1:
    opt_groups   = sorted([x for x in res["group"].dropna().unique().tolist() if x])
    sel_groups   = st.multiselect("Gruppe (page_name)", options=opt_groups, default=[])

with fcol2:
    opt_sponsors = sorted([x for x in res["sponsor"].dropna().unique().tolist() if x])
    sel_sponsors = st.multiselect("Sponsoren (bylines)", options=opt_sponsors, default=[])

with fcol3:
    opt_ads      = res["id"].astype(str).unique().tolist()
    sel_ads      = st.multiselect("Ad", options=opt_ads, default=[])

with fcol4:
    st.button("Filter zurücksetzen", use_container_width=True, key="reset_filters")


mask = pd.Series(True, index=res.index)
if sel_groups:
    mask &= res["group"].isin(sel_groups)
if sel_sponsors:
    mask &= res["sponsor"].isin(sel_sponsors)
if sel_ads:
    mask &= res["id"].astype(str).isin(sel_ads)

res_f = res[mask].copy()
if res_f.empty:
    st.info("Keine Ergebnisse für die aktuelle Filterkombination.")
    st.stop()

# Für Vorschau je Ad eine Zeile
preview_df = res_f.drop_duplicates(subset=["id"]).copy()

# --- Helper: kurze Begründung aus den Kategorie-Rationales bauen ---
def _short_reason(row: pd.Series) -> str:
    cats = [
        "factual_accuracy", "framing_quality", "visual_mislead", "targeting_risks",
        "policy_legal", "transparency_context", "consistency_history", "usability_accessibility",
    ]
    bits = []
    for k in cats:
        v = row.get(f"rationale_{k}")
        if isinstance(v, str) and v.strip():
            bits.append(v.strip())
    return " • ".join(bits[:3])

def _pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    cat_map = {
        "factual_accuracy": "Faktischer Gehalt",
        "framing_quality": "Framing",
        "visual_mislead": "Visuals",
        "targeting_risks": "Targeting",
        "policy_legal": "Policy/Recht",
        "transparency_context": "Transparenz",
        "consistency_history": "Konsistenz",
        "usability_accessibility": "Usability/Barrierefreiheit",
    }
    ren = {"id": "Ad-ID", "overall_risk": "Gesamtrisiko", "overall_confidence": "Gesamt-Konfidenz"}
    for k, nice in cat_map.items():
        if f"score_{k}" in df.columns:      ren[f"score_{k}"]      = f"Score – {nice}"
        if f"confidence_{k}" in df.columns: ren[f"confidence_{k}"] = f"Konf. – {nice}"
        if f"rationale_{k}" in df.columns:  ren[f"rationale_{k}"]  = f"Begründung – {nice}"
        if f"examples_{k}" in df.columns:   ren[f"examples_{k}"]   = f"Beispiele – {nice}"
    return df.rename(columns=ren)

# Kurzbegründung und hübsche Spaltenlabels
preview_df["Kurzbegründung"] = preview_df.apply(_short_reason, axis=1)
preview_pretty = _pretty_columns(preview_df.copy())

st.subheader("Ergebnisvorschau")
cols_scores = [
    "Score – Faktischer Gehalt","Score – Framing","Score – Visuals",
    "Score – Targeting","Score – Policy/Recht","Score – Transparenz",
    "Score – Konsistenz","Score – Usability/Barrierefreiheit"
]
show_cols = ["Ad-ID","Gesamtrisiko","Gesamt-Konfidenz","Kurzbegründung"] + [c for c in cols_scores if c in preview_pretty.columns]
st.dataframe(preview_pretty[show_cols], use_container_width=True)

# -------------------- Pro-Ad Detail --------------------
st.subheader("Details je Ad")
ordered_ids = preview_df["id"].astype(str).tolist()

for ad_id in ordered_ids:
    base_row = df_out[df_out["id"].astype(str) == ad_id].iloc[0]
    head = f"Ad {ad_id} — Gesamtrisiko {float(base_row.get('overall_risk') or 0):.2f} · Konf. {float(base_row.get('overall_confidence') or 0):.2f}"
    with st.expander(head, expanded=False):
        cats = [
            ("Faktischer Gehalt", "factual_accuracy"),
            ("Framing", "framing_quality"),
            ("Visuals", "visual_mislead"),
            ("Targeting", "targeting_risks"),
            ("Policy/Recht", "policy_legal"),
            ("Transparenz", "transparency_context"),
            ("Konsistenz", "consistency_history"),
            ("Usability/Barrierefreiheit", "usability_accessibility"),
        ]
        chart_rows, table_rows = [], []
        for nice, key in cats:
            sc  = float(base_row.get(f"score_{key}") or 0.0)
            cf  = float(base_row.get(f"confidence_{key}") or 0.0)
            rat = base_row.get(f"rationale_{key}") or ""
            ex  = base_row.get(f"examples_{key}") or []
            chart_rows.append({"Kategorie": nice, "Score": sc})
            table_rows.append({
                "Kategorie": nice, "Score": sc, "Konf.": cf,
                "Begründung": rat, "Beispiele": "; ".join(map(str, ex))
            })
        fig = px.bar(pd.DataFrame(chart_rows), x="Kategorie", y="Score", title=None)
        fig.update_layout(height=260, margin=dict(l=10, r=10, b=10, t=10))

        # eindeutige Keys pro Ad
        st.plotly_chart(fig, use_container_width=True, key=f"weak_bar_{ad_id}")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, key=f"weak_tbl_{ad_id}")
