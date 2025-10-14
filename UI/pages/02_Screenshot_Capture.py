# UI/pages/02_Screenshot_Capture.py
import os
import sys
import json
import time
from io import BytesIO
from pathlib import Path
from psycopg import sql
from datetime import date
import streamlit as st
import pandas as pd
from PIL import Image, ImageStat   # für Heuristik „Video-Fehler“-Banner

# Fortschrittsdatei (Agent schreibt hier hinein)
PROGRESS_PATH = Path(__file__).resolve().parents[2] / ".run" / "screenshot_progress.json"
PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- DB Zugriff (read-only) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

st.set_page_config(page_title="Screenshots (Meta Ad Library)", page_icon="📸", layout="wide")
st.title("📸 Screenshots (Meta Ad Library)")

# ------------------------------------------------------------------
# Postgres-Session einstellen (kurze Timeouts, Name)
# ------------------------------------------------------------------
def _configure_pg_session(conn, app_name: str) -> None:
    try:
        conn.autocommit = False
    except Exception:
        pass
    try:
        cur = conn.cursor()
        cur.execute(sql.SQL("SET application_name = {}").format(sql.Literal(app_name)))
        cur.execute("SET lock_timeout = '2s'")
        cur.execute("SET idle_in_transaction_session_timeout = '2min'")
        cur.execute("SET statement_timeout = '5min'")
        cur.close()
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

def _get_conn():
    conn = st.session_state.get("conn")
    if conn is None:
        conn = connect()
        _configure_pg_session(conn, "ui_screenshot_capture")
        st.session_state["conn"] = conn
    return conn

def _get_internal_ids(conn, campaign_name: str | None = None, limit: int = 600) -> list[int]:
    """
    Holt ad.id’s, sortiert nach 'jüngste media-Einträge zuerst'.
    """
    cur = conn.cursor()
    if campaign_name:
        cur.execute(
            """
            SELECT a.id
            FROM ads a
            JOIN campaigns c ON c.id = a.campaign_id
            LEFT JOIN media m ON m.ad_id = a.id
            WHERE c.name = %s
            GROUP BY a.id
            ORDER BY max(m.created_at) DESC NULLS LAST
            LIMIT %s
            """,
            (campaign_name, limit),
        )
    else:
        cur.execute(
            """
            SELECT a.id
            FROM ads a
            LEFT JOIN media m ON m.ad_id = a.id
            GROUP BY a.id
            ORDER BY max(m.created_at) DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        )
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    return ids

# einmalig erzeugen / cachen
conn = _get_conn()
campaign_name = st.session_state.get("campaign_name")  # optional
internal_ids = st.session_state.get("internal_ids")
if not internal_ids:
    internal_ids = _get_internal_ids(conn, campaign_name=campaign_name, limit=600)
    st.session_state["internal_ids"] = internal_ids

# -------------------------------------------------
# Hilfsfunktionen (JSON finden / IDs mappen)
# -------------------------------------------------
def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "agents" / "capture_ad_snapshot.py"

def _today_str() -> str:
    return date.today().isoformat()

def _json_from_snapshot_table(conn) -> Path | None:
    cur = conn.cursor()
    sqls = [
        "SELECT file_url FROM api_snapshots WHERE snapshot_date = CURRENT_DATE ORDER BY created_at DESC LIMIT 1",
        "SELECT file_url FROM snapshots    WHERE snapshot_date = CURRENT_DATE ORDER BY created_at DESC LIMIT 1",
    ]
    file_url = None
    for s in sqls:
        try:
            cur.execute(s)
            row = cur.fetchone()
            if row and row[0]:
                file_url = row[0]
                break
        except Exception:
            continue
    cur.close()
    if not file_url:
        return None
    try:
        p = Path(str(file_url).replace("file://", "")) if str(file_url).startswith("file://") else Path(str(file_url))
    except Exception:
        return None
    return p if p.exists() else None

def _find_todays_json_fallback() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    today = _today_str()
    for p in root.rglob(f"{today}.json"):
        return p
    return None

def _load_ad_external_ids_from_json(json_path: Path) -> list[str]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    src = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else [])
    ads = []
    for r in src:
        aid = str(r.get("id") or "").strip()
        url = r.get("ad_snapshot_url")
        if aid and isinstance(url, str) and url.startswith("http"):
            ads.append(aid)
    return ads

def _map_external_to_internal_ids(conn, ad_ids_external: list[str]) -> dict[str, int]:
    if not ad_ids_external:
        return {}
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, ad_external_id
        FROM ads
        WHERE ad_external_id = ANY(%s)
        """,
        (ad_ids_external,),
    )
    rows = cur.fetchall()
    cur.close()
    mapping = {}
    for iid, ext in rows:
        mapping[str(ext)] = int(iid)
    return mapping

def _count_ads_with_screenshot(conn, internal_ids: list[int]) -> int:
    if not internal_ids:
        return 0
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT ad_id) FROM media WHERE ad_id = ANY(%s)", (internal_ids,))
    n = cur.fetchone()[0] or 0
    cur.close()
    return int(n)

def _fetch_latest_media_by_ad_ids(conn, ad_ids, limit_per_ad=1) -> pd.DataFrame:
    cur = conn.cursor()
    # vorhandene Pfadspalte erkennen
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'media'
          AND column_name IN ('local_path','path','file_url','file_path')
    """)
    cols = {r[0] for r in cur.fetchall()}
    if   "local_path" in cols: path_expr = "m.local_path"
    elif "path"       in cols: path_expr = "m.path"
    elif "file_path"  in cols: path_expr = "m.file_path"
    elif "file_url"   in cols: path_expr = "m.file_url"
    else:                      path_expr = "''::text"

    sql = f"""
    WITH latest AS (
      SELECT
        m.id AS media_id,
        m.ad_id,
        m.kind,
        m.format,
        {path_expr} AS local_path,
        m.date_folder,
        m.created_at,
        a.ad_external_id,
        -- neu: jüngster page_name via LATERAL aus ad_llm_fused
        (ff.fused->'api'->'raw'->>'page_name') AS page_name,
        ROW_NUMBER() OVER (PARTITION BY m.ad_id ORDER BY m.created_at DESC) AS rn
      FROM media m
      JOIN ads a ON a.id = m.ad_id
      LEFT JOIN LATERAL (
        SELECT fused
        FROM ad_llm_fused f
        WHERE f.ad_id = m.ad_id
        ORDER BY f.snapshot_date DESC, f.created_at DESC
        LIMIT 1
      ) ff ON TRUE
      WHERE m.ad_id = ANY(%s)
        AND (m.format IS NULL OR m.format <> 'video_error')
    )
    SELECT ad_id, media_id, kind, format, local_path, date_folder, created_at,
           ad_external_id, page_name
    FROM latest
    WHERE rn <= %s
    ORDER BY ad_id, created_at DESC;
    """
    cur.execute(sql, (ad_ids, int(limit_per_ad)))
    rows = cur.fetchall()
    cur.close()

    df = pd.DataFrame(rows, columns=[
        "ad_id","media_id","kind","format","local_path","date_folder","created_at",
        "ad_external_id","page_name"
    ])

    def _norm_local_path(p: str | None) -> str:
        if not p:
            return ""
        s = str(p)
        return s[7:] if s.startswith("file://") else s

    if not df.empty:
        df["local_path"] = df["local_path"].map(_norm_local_path)

    return df

def _load_base64_for_media(conn, media_id: int) -> tuple[str | None, str | None]:
    """
    Lädt (mime_type, b64) für genau eine media_id.
    Wird nur beim Rendern der ausgewählten Medien aufgerufen.
    """
    cur = conn.cursor()
    cur.execute("SELECT mime_type, b64 FROM media_base64 WHERE media_id = %s", (int(media_id),))
    row = cur.fetchone()
    cur.close()
    if not row:
        return None, None
    return row[0], row[1]

# ---------- Heuristik: „Video kann nicht abgespielt werden“-Banner ----------
def _has_video_error_band_on_disk(path: str) -> bool:
    """Erkennt den grauen Fehlerbalken am unteren Rand des Screenshots – arbeitet auf Datei, nicht Base64."""
    if not path or not os.path.exists(path):
        return False
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return False

    w, h = im.size
    if w < 300 or h < 220:
        return False

    y0 = int(h * 0.82)                     # untere ~18%
    roi = im.crop((0, y0, w, h))
    stat = ImageStat.Stat(roi)
    mean = sum(stat.mean) / 3.0
    var  = sum(stat.var)  / 3.0

    if not (85 <= mean <= 175):            # grauer Balken
        return False
    if var > 350:                          # zu unruhig
        return False

    # klein wenig „weiß“ (Text)
    px = roi.load()
    white = 0
    step = max(4, min(w, h)//160)
    for y in range(0, roi.size[1], step):
        for x in range(0, roi.size[0], step):
            r,g,b = px[x,y]
            if r > 228 and g > 228 and b > 228:
                white += 1
    total = (roi.size[0]//step) * (roi.size[1]//step)
    return (white / max(1,total)) > 0.002

# -------------------------------------------------
# Beschreibung
# -------------------------------------------------
with st.expander("Wie funktioniert das?", expanded=True):
    st.markdown(
        """
        **Start** erstellt (headless, Playwright) Screenshots zu den **heutigen Ad-IDs** aus der
        gespeicherten JSON (Snapshot von heute) und schreibt sie direkt in die DB
        (`media` + `media_base64`). Alle Vorgänge werden automatisch erkannt – die Anzeige
        aktualisiert sich **alle 5 s**, solange der Lauf aktiv ist.
        """
    )

# -------------------------------------------------
# DB verbinden + heutige JSON bestimmen
# -------------------------------------------------
try:
    conn = connect()
except Exception as e:
    conn = None
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")

json_path: Path | None = None
ad_ids_external: list[str] = []
ad_ext_to_int: dict[str,int] = {}
internal_ids: list[int] = []

json_source_note = ""
if conn:
    json_path = _json_from_snapshot_table(conn)
    if not json_path:
        json_path = _find_todays_json_fallback()
        json_source_note = " (Fallback: Dateisuche)"
    else:
        json_source_note = " (aus DB-Snapshot)"

    if json_path and json_path.exists():
        ad_ids_external = _load_ad_external_ids_from_json(json_path)
        ad_ext_to_int = _map_external_to_internal_ids(conn, ad_ids_external)
        internal_ids = list(ad_ext_to_int.values())

with st.container(border=True):
    if json_path:
        st.markdown(
            f"**Verwendete JSON**: `{json_path}`{json_source_note}  "
            f"• **Ad-IDs heute**: {len(ad_ids_external)}  "
            f"• **davon in DB vorhanden**: {len(internal_ids)}"
        )
    else:
        st.warning("Keine heutige JSON gefunden (weder Snapshot in DB noch Fallback auf der Platte).")

# -------------------------------------------------
# Start / Stop + EIN Status-Placeholder
# -------------------------------------------------
status_box = st.empty()  # hier erscheint „Lauf gestartet/gestoppt“

c1, c2 = st.columns([1,1])
with c1:
    start_disabled = not (json_path and json_path.exists() and st.session_state.get("shot_pid") is None)
    if st.button("▶️ Screenshots starten", disabled=start_disabled, type="primary"):
        script_path = _script_path()
        if not script_path.exists():
            st.error(f"Skript nicht gefunden: {script_path}")
            st.stop()

        out_root = json_path.parent
        args = [
            sys.executable, str(script_path),
            str(json_path), str(out_root),
            "--progress-file", str(PROGRESS_PATH)  # <<< wichtig für Live-Fortschritt
        ]

        creationflags = 0
        if os.name == "nt":
            CREATE_NO_WINDOW = 0x08000000
            creationflags = CREATE_NO_WINDOW

        try:
            import subprocess
            proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            st.session_state["shot_pid"] = proc.pid
            status_box.success("Lauf gestartet.")
        except Exception as e:
            st.error(f"Konnte den Prozess nicht starten: {e}")

with c2:
    stop_disabled = st.session_state.get("shot_pid") is None
    if st.button("⏹️ Lauf abbrechen", disabled=stop_disabled):
        pid = st.session_state.get("shot_pid")
        if pid:
            try:
                import subprocess
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                else:
                    os.kill(pid, 15)
            except Exception:
                pass
        st.session_state["shot_pid"] = None
        status_box.info("Prozess gestoppt.")

# -------------------------------------------------
# Fortschritt (aus Progress-JSON, Fallback DB)
# -------------------------------------------------
running = st.session_state.get("shot_pid") is not None
n_total = len(ad_ids_external)

proc = ins = tot = last = None
try:
    data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    proc = int(data.get("processed") or 0)
    ins  = int(data.get("inserted")  or 0)
    tot  = int(data.get("total")     or 0) or n_total
    last = data.get("last_ad_id")
except Exception:
    pass

if ins is None and conn and internal_ids:
    ins = _count_ads_with_screenshot(conn, internal_ids)
if tot is None:
    tot = n_total

pg_col, info_col = st.columns([3,2])
with pg_col:
    if tot:
        pct = min(1.0, (ins or 0) / tot)
        st.progress(pct, text=f"DB gespeichert: {ins or 0} / {tot} · verarbeitet: {proc or 0}/{tot}")
    else:
        st.info("Noch kein Fortschritt erkannt.")
with info_col:
    if running:
        st.info("Lauf aktiv. Anzeige aktualisiert sich automatisch (alle 5 s).")
        if last:
            st.caption(f"Letzte Ad-ID: {last}")
    else:
        st.caption("Kein aktiver Lauf.")

# -------------------------------------------------
# Anzeige: neuester Screenshot je ad_id (+ Filter)
# -------------------------------------------------
st.subheader("Heutige Screenshots (neuester je media_id)")

# Toggle (zusätzlicher Datei-basierter Filter)
hide_video_errors = st.checkbox("Video-Fehler-Shots ausblenden", value=False)

df_media = pd.DataFrame()
if conn and internal_ids:
    df_media = _fetch_latest_media_by_ad_ids(conn, internal_ids, limit_per_ad=1)

    # zusätzlicher Datei-basierter Filter (Agent setzt format='video_error' bereits in SQL raus)
    if hide_video_errors and not df_media.empty:
        marks = []
        for p in df_media["local_path"].fillna(""):
            ok = False
            try:
                if p and os.path.exists(p):
                    # leichter Datei-Check (keine Base64-Massen)
                    from PIL import Image, ImageStat
                    im = Image.open(p).convert("RGB")
                    w, h = im.size
                    if w >= 300 and h >= 220:
                        y0 = int(h * 0.82)
                        roi = im.crop((0, y0, w, h))
                        stat = ImageStat.Stat(roi)
                        mean = sum(stat.mean)/3.0
                        var = sum(stat.var)/3.0
                        ok = (85 <= mean <= 175) and (var <= 350)
                # wenn kein Pfad vorhanden -> wir können hier nicht sicher erkennen -> NICHT filtern
            except Exception:
                ok = False
            marks.append(ok)
        # filtere nur dort, wo wir den Balken sicher erkannt haben
        if marks:
            df_media = df_media[~pd.Series(marks).values]

if df_media.empty:
    st.info("Noch keine passenden Screenshots gefunden. (Es wird je ad_id der **neueste** Screenshot gezeigt.)")
else:
    # Label: "Gruppe (external_id)"
    def _make_label(r):
        page = (str(r.get("page_name") or "").strip()) or "Unbekannte Gruppe"
        ext  = str(r.get("ad_external_id") or "").strip() or "-"
        return f"{page} ({ext})"

    df_media["label"] = df_media.apply(_make_label, axis=1)

    # Mapping Label -> media_id
    label2mid = dict(zip(df_media["label"], df_media["media_id"]))

    labels_sorted = df_media["label"].tolist()

    left, right = st.columns(2)
    with left:
        sel_label_left = st.selectbox("Anzeige links (Gruppe · external_id)", options=labels_sorted, index=0)
    with right:
        idx2 = 1 if len(labels_sorted) > 1 else 0
        sel_label_right = st.selectbox("Anzeige rechts (Gruppe · external_id)", options=labels_sorted, index=idx2)

    def _render_by_label(col, label):
        media_id_val = label2mid[label]
        row = df_media[df_media["media_id"] == media_id_val].iloc[0]
        cap = (
            f"{row.get('page_name') or 'Unbekannte Gruppe'} ({row.get('ad_external_id') or '-'})  ·  "
            f"media_id: {row['media_id']}  ·  ad_id: {row['ad_id']}  ·  {row['created_at']:%Y-%m-%d %H:%M}"
        )
        p = (row.get("local_path") or "").strip()
        if p and os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)
        else:
            mime, b64 = _load_base64_for_media(conn, int(row["media_id"]))
            if mime and b64:
                col.image(f"data:{mime};base64,{b64}", caption=cap, use_container_width=True)
            else:
                col.info("Dateipfad/DB-Base64 nicht vorhanden.")

    colA, colB = st.columns(2)
    _render_by_label(colA, sel_label_left)
    _render_by_label(colB, sel_label_right)
    
# Reconnect-Button
if st.button("🔄 DB-Reconnect"):
    try:
        st.session_state.pop("conn")
    except KeyError:
        pass
    st.rerun()

# Auto-Refresh während Lauf
if running:
    time.sleep(5)
    st.rerun()
