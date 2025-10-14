# pages/99_LLM_Analyse_Screenshots.py
# -*- coding: utf-8 -*-
import os, sys, json, time, textwrap
from pathlib import Path
from typing import Tuple

import streamlit as st
import pandas as pd

# ---------- API-Key Check ----------
def _check_api_keys(provider_ui: str) -> bool:
    if provider_ui == "OpenAI":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider_ui == "Gemini":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    return False

# ---------- DB / Helpers ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / ".run"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_PATH = RUN_DIR / "llm_analysis_progress.json"
ADIDS_TODAY_PATH = RUN_DIR / "llm_ad_ids_today.json"
PROMPT_OVERRIDE_PATH = RUN_DIR / "llm_prompt_override.txt"
LOG_PATH = RUN_DIR / "llm_analysis.log"

st.set_page_config(page_title="LLM-Analyse (Screenshots)", page_icon="🧠", layout="wide")
st.title("🧠 LLM-Analyse der Screenshots")

# ---------- DB helpers ----------
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
        if str(file_url).startswith("file://"):
            p = Path(str(file_url).replace("file://", ""))
        else:
            p = Path(str(file_url))
    except Exception:
        return None
    return p if p.exists() else None

def _find_todays_json_fallback() -> Path | None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for p in (ROOT / "Eigenmietwert").rglob(f"{today}.json"):
        return p
    for p in ROOT.rglob(f"{today}.json"):
        return p
    return None

def _load_ad_external_ids_from_json(json_path: Path) -> list[str]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        src = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        src = data["data"]
    else:
        src = []
    out = []
    for r in src:
        aid = str(r.get("id") or "").strip()
        url = r.get("ad_snapshot_url")
        if aid and isinstance(url, str) and url.startswith("http"):
            out.append(aid)
    return out

def _map_external_to_internal_ids(conn, ad_ids_external: list[str]) -> dict[str, int]:
    if not ad_ids_external:
        return {}
    cur = conn.cursor()
    cur.execute(
        "SELECT id, ad_external_id FROM ads WHERE ad_external_id = ANY(%s)",
        (ad_ids_external,),
    )
    rows = cur.fetchall()
    cur.close()
    return {str(ext): int(iid) for (iid, ext) in rows}

# ---------- Agent script paths ----------
def _script_path(provider: str) -> Path:
    base = ROOT / "agents" / "LLM-Initial-Analysis"
    if provider == "openai":
        return base / "openai_image_analysis.py"
    if provider == "gemini":
        return base / "gemini_image_analysis.py"
    raise ValueError("Nur OpenAI oder Gemini erlaubt.")

# ---------- Results helpers ----------
def _fetch_results(conn, campaign_slug: str, provider_ui: str, limit: int = 500):
    """
    Liefert gespeicherte Ergebnisse inkl. page_name (neuester fused via LATERAL).
    """
    provider = {"OpenAI": "openai", "Gemini": "gemini"}.get(provider_ui, str(provider_ui).lower())
    sql = """
      SELECT
        ar.id              AS analysis_result_id,
        ar.created_at,
        ar.result          AS json_result,
        ad.id              AS ad_id,
        ad.ad_external_id  AS ad_external_id,
        (ff.fused->'api'->'raw'->>'page_name') AS page_name,
        m_latest.id        AS media_id,
        mb.b64,
        mb.mime_type
      FROM analysis_results ar
      JOIN analyses a      ON a.id = ar.analysis_id
      JOIN ads ad          ON ad.id = ar.ad_id
      JOIN campaigns c     ON c.id = ad.campaign_id
      LEFT JOIN LATERAL (
        SELECT fused
        FROM ad_llm_fused f
        WHERE f.ad_id = ad.id
        ORDER BY f.snapshot_date DESC, f.created_at DESC
        LIMIT 1
      ) ff ON TRUE
      LEFT JOIN LATERAL (
        SELECT m.*
        FROM media m
        WHERE m.ad_id = ad.id AND m.kind = 'screenshot'
        ORDER BY m.created_at DESC
        LIMIT 1
      ) m_latest ON TRUE
      LEFT JOIN media_base64 mb ON mb.media_id = m_latest.id
      WHERE c.slug = %s
        AND a.provider = %s::provider_kind
      ORDER BY ar.created_at DESC
      LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug, provider, int(limit)))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df

def _fetch_media_b64_for_result(conn, result_json: dict, ad_id: int) -> Tuple[str|None, str|None, str]:
    media_id = None
    try:
        sid = result_json.get("screenshot_id") or ""
        if isinstance(sid, str) and sid.startswith("media:"):
            media_id = int(sid.split(":",1)[1])
    except Exception:
        media_id = None

    cur = conn.cursor()
    if media_id:
        cur.execute("""
            SELECT mb.b64, mb.mime_type, m.created_at
            FROM media m
            LEFT JOIN media_base64 mb ON mb.media_id = m.id
            WHERE m.id = %s
            """, (media_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            b64, mime, ts = row
            cap = f"media_id: {media_id} · {ts:%Y-%m-%d %H:%M}"
            return b64, mime, cap

    cur = conn.cursor()
    cur.execute("""
        SELECT m.id, mb.b64, mb.mime_type, m.created_at
        FROM media m
        LEFT JOIN media_base64 mb ON mb.media_id = m.id
        WHERE m.ad_id = %s AND m.kind = 'screenshot'
        ORDER BY m.created_at DESC
        LIMIT 1
        """, (ad_id,))
    row = cur.fetchone()
    cur.close()
    if row:
        mid, b64, mime, ts = row
        cap = f"media_id: {mid} · {ts:%Y-%m-%d %H:%M}"
        return b64, mime, cap
    return None, None, "Kein Screenshot gefunden"

# ---------- Defaults ----------
DEFAULT_PROMPT = textwrap.dedent("""\
Du analysierst Screenshots von Social-Media-Werbeanzeigen (Base64-Bilder).
Vorgehen:

Qualitätscheck:
Nur dann als "geeignet": false markieren, wenn EINDEUTIG ein Fehler/Platzhalter-Overlay den Hauptinhalt verdeckt,
z. B. klarer grauer Banner mit Text wie „Dieses/Leider kann dieses Video nicht abgespielt/angezeigt werden“,
„Mehr dazu“-Hinweis als Fehlerbanner oder vergleichbar dominantes Fehler-Overlay.
Wenn der Anzeigentext/Anzeige-Layout erkennbar ist, DANN TROTZDEM analysieren (geeignet=true).

Wenn kein eindeutiger Fehlerbanner erkennbar ist oder Zweifel bestehen: IMMER als geeignet=true werten und vollständig analysieren.

Output: EIN einziges JSON-Objekt exakt nach Schema. Keine Kommentare/Markdown. Bei geeignet=false ist "analyse": null.

— Zusätzliche Analyseanforderungen (ergänzend, ohne Schemaänderung) —
• Alle sichtbaren Elemente im Bild systematisch erfassen:
– Typ, Position, Farbe, Form, Größe, Stil, Funktion, Bedeutung
– CTA-Buttons, Icons, Rabatt-Sticker, Social-Media-UI (Like/Share/Comment, Reaktionen, Menüs, Scrollleisten), Logos, Personen, Produkte, Layout-Raster
– Für jedes Element: Was ist es? Wo ist es? Wozu dient es? Wie wirkt es?

• Textebene vollständig analysieren:
– Alle Textblöcke extrahieren (inkl. UI-Texte, Kommentare, Randnotizen)
– Pro Textblock: Inhalt (Wortlaut), Funktion (CTA/Branding/Info/Rabatt usw.), Sprachebene (formell/werblich/neutral …), Ton & Wirkung (motiviert/informiert/drängt …)

• Quantitative Textmetriken bestimmen (in die bestehenden Felder einspeisen):
– Zeichen- & Wortanzahl (Headline und gesamt), durchschnittliche Wortlänge
– Anzahl unterschiedlicher Schriftarten
– Verhältnis Textfläche/Bildfläche (in %)

• Screenshot-Erkennung & Plattformkontext:
– Prüfen, ob es sich um einen Screenshot handelt
– UI-Indikatoren dokumentieren (Like-Zähler, Kommentare, Buttons, Menüs, Scrollbars)
– Mögliche Plattform angeben (Facebook/Instagram/LinkedIn/TikTok/Google/Unbekannt)

• Visuelle Gestaltung & Layoutanalyse:
– Farbkontraste und dominante Farben (Farbcodes)
– Kompositionstyp (zentral/asymmetrisch/Raster)
– Blickführung (zentral/dynamisch/radial)
– Layoutstruktur (Social-Feed/Kachel/Story/klassisch)
– Verhältnis Text/Bild/Weißraum (in %)
– Schriftarten & -größenverteilung, Textausrichtung
– Professionalitätsgrad des Designs qualitativ beurteilen (in bestehende Felder beschreibend abbilden)

• Semantische & persuasive Strategie:
– Emotionale/rationale Appelle
– Symbole (Haken/Herz/Stern/Flamme …) und deren Wirkung (Vertrauen/Dringlichkeit …)
– Werbeversprechen-Typ (USP/ESP/generisch)
– Zielgruppenindikatoren (Bildsprache, Vokabular)
– Framing (Gewinn/Verlust/Moral/Autorität/sozialer Vergleich)
– Ansprache-Typ (direkt/allgemein/duzend/siezend)

• Wenn Informationen nicht erkennbar sind, verwende "Unklar" oder false; verändere niemals die JSON-Struktur oder Schlüsselnamen.

JSON-Schema (exakt einhalten):
{
"kampagne_id": "[KAMPAGNEN-ID]",
"analysetool": "OpenAI",
"analyse_datum": "[YYYY-MM-DD]",
"screenshot_id": "[SCREENSHOT-ID-ODER-HASH]",
"geeignet": true,
"analyse": {
"visuelle_features": {
"farbpalette": ["#FFAA00", "#000000", "#FFFFFF"],
"schriftarten_erkannt": ["Arial", "Sans Serif"],
"schriftgrößen_verteilung": { "klein": 2, "mittel": 1, "groß": 1 },
"textausrichtung": "zentriert | linksbündig | rechtsbündig | gemischt",
"flächenverteilung": { "textfläche": 23, "bildfläche": 60, "weißraum": 17 },
"kompositionstyp": "Zentrumskomposition | asymmetrisch | Raster",
"bildtyp": "Foto | Illustration | CGI | Stock | Screenshot",
"blickführung": "zentral | dynamisch | radial",
"salienzverteilung": 0.0,
"dominante_layoutstruktur": "Einspaltig | mehrspaltig | Social-Feed | Werbekachel | klassisch",
"plattformkontext_erkannt": false,
"plattform": "Unbekannt",
"elemente": [
{
"element": "Text",
"position": "Zentrum",
"farbe": "Unklar",
"größe": "mittel",
"form": "rechteckig",
"interaktiv_erscheinung": false,
"funktion": "Textblock",
"bedeutung": "Unklar",
"inhalt": "Unklar",
"person_mimik_erkannbar": "nicht sichtbar",
"bild_inhalt": "Unklar",
"markenlogo_erkannt": false
}
]
},
"textuelle_features": {
"headline_länge": "Unklar",
"headline_zeichenanzahl": 0,
"headline_wortanzahl": 0,
"gesamtzeichenanzahl": 0,
"gesamtwortanzahl": 0,
"durchschnittliche_wortlänge": 0.0,
"anzahl_textblöcke": 0,
"anzahl_schriftarten": 0,
"text_bild_verhältnis": 0.0,
"cta_typ": "Unklar",
"cta_position": "nicht vorhanden",
"cta_visuelle_prominenz": "gering",
"cta_wirkungseinschätzung": "schwach",
"sprachstil": "Unklar",
"tonalität": "Unklar",
"textgliederung_erkennbar": false,
"wortartenverteilung": { "Substantive": 0, "Verben": 0, "Adjektive": 0, "Pronomen": 0 },
"text_inhalte": []
},
"semantische_features": {
"argumenttyp": "Unklar",
"bild_text_verhältnis": "Unklar",
"symbolgebrauch": {
"symbol_erkannt": false,
"symbol_typ": "Unklar",
"symbol_bedeutung": "Unklar"
},
"werbeversprechen": "generisch",
"zielgruppe": "Allgemein",
"zielgruppen_indikatoren": [],
"emotionaler_apell": "Unklar",
"framing_typ": "Unklar",
"ansprache_typ": "Unklar"
}
}
}

Wenn das Bild ungeeignet ist, liefere:
{
"kampagne_id": "[KAMPAGNEN-ID]",
"analysetool": "OpenAI",
"analyse_datum": "[YYYY-MM-DD]",
"screenshot_id": "[SCREENSHOT-ID]",
"geeignet": false,
"analyse": null
}

Antworte AUSSCHLIESSLICH mit genau einem JSON-Objekt, beginne mit { und ende mit }. Keine Einleitung, kein Markdown, keine Kommentare.
""").strip()

# ---------- UI: Auswahl ----------
# (Campaign Slug wird nicht mehr editiert im UI – wir nehmen Session-Wert oder Default.)
campaign_slug = st.session_state.get("campaign_slug", "eigenmietwert")

colA, colB = st.columns([1,1])
with colA:
    provider_label = st.radio("Provider", options=["OpenAI", "Gemini"], index=0, horizontal=True)
    provider = "openai" if provider_label == "OpenAI" else "gemini"
with colB:
    limit_target = st.number_input("Ziel: Anzahl Analysen speichern", 1, 1000, 50, 5)

# Prompt-Konfiguration
with st.expander("📝 Prompt konfigurieren", expanded=False):
    st.caption("Du kannst den Standard-Prompt einsehen/überschreiben. Der Text wird für **neue** Analysen verwendet.")
    use_custom_prompt = st.checkbox("Eigenen Prompt verwenden", value=False)
    prompt_text = st.text_area("Prompt", value=DEFAULT_PROMPT, height=180, disabled=not use_custom_prompt)
    if use_custom_prompt and st.button("Prompt speichern (für nächsten Lauf)"):
        PROMPT_OVERRIDE_PATH.write_text(prompt_text or DEFAULT_PROMPT, encoding="utf-8")
        st.success("Prompt-Override gespeichert.")
    elif not use_custom_prompt:
        try:
            if PROMPT_OVERRIDE_PATH.exists():
                PROMPT_OVERRIDE_PATH.unlink()
                st.info("Prompt-Override deaktiviert.")
        except Exception:
            pass

# OpenAI/Gemini Modell
if provider == "openai":
    model = st.selectbox("OpenAI-Modell", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
else:
    model = st.selectbox("Gemini-Modell", ["gemini-1.5-pro","gemini-1.5-flash"], index=0)

with st.expander("Erweitert"):
    fetch_n = st.number_input("Max. Screenshots aus DB ziehen (fetch)", 10, 5000, 400, 50)
    analyze_all = st.checkbox("Alle analysieren (keine Ungeeignet-Filter)", value=True)
    st.markdown("**Rate-Limit / Backoff**")
    rl_cols = st.columns(6)
    with rl_cols[0]:
        rl_tpm_headroom = st.number_input("TPM-Headroom", 0, 100000, 20000, 1000, help="Abzug von deinem TPM-Limit.")
    with rl_cols[1]:
        rl_rpm_headroom = st.number_input("RPM-Headroom", 0, 10000, 50, 5, help="Abzug von deinem RPM-Limit.")
    with rl_cols[2]:
        rl_conc = st.number_input("Max. Concurrency", 1, 64, 4, 1, help="Parallele Requests (vom Agent genutzt).")
    with rl_cols[3]:
        rl_sleep = st.number_input("Base sleep (ms)", 0, 5000, 800, 50, help="Grundwartezeit vor Retries.")
    with rl_cols[4]:
        rl_retries = st.number_input("Max. Retries", 0, 10, 5, 1)
    with rl_cols[5]:
        rl_jitter = st.number_input("Jitter (ms)", 0, 2000, 250, 50, help="Zufälliger Zusatz pro Retry.")

st.divider()

# ---------- Heutige ad_ids bestimmen ----------
conn = None
try:
    conn = connect()
except Exception as e:
    st.error(f"DB-Verbindung fehlgeschlagen: {e}")

ad_ids_external, ad_ext_to_int, internal_ids = [], {}, []
json_path, src_note = None, ""
if conn:
    jp_db = _json_from_snapshot_table(conn)
    json_path = jp_db or _find_todays_json_fallback()
    src_note = " (aus DB-Snapshot)" if jp_db else " (Fallback: Dateisuche)"
    if json_path and json_path.exists():
        ad_ids_external  = _load_ad_external_ids_from_json(json_path)
        ad_ext_to_int    = _map_external_to_internal_ids(conn, ad_ids_external)
        internal_ids     = list(ad_ext_to_int.values())
        ADIDS_TODAY_PATH.write_text(json.dumps(internal_ids), encoding="utf-8")

with st.container(border=True):
    if json_path:
        st.markdown(
            f"**Heutige Abfrage**: `{json_path}`{src_note}  "
            f"• **ext. Ad-IDs**: {len(ad_ids_external)}  "
            f"• **in DB gemappt**: {len(internal_ids)}"
        )
    else:
        st.warning("Heute keine JSON gefunden – ohne diese Liste wird **keine** Analyse gestartet.")

# ---------- Start / Stop ----------
st.session_state.setdefault("llm_pid", None)
st.session_state.setdefault("llm_running", False)
st.session_state.setdefault("did_done_reload", False)

c1, c2 = st.columns([1,1])
with c1:
    disabled = (st.session_state["llm_pid"] is not None) or (not campaign_slug.strip()) or (not internal_ids)
    if st.button("▶️ Analyse starten", type="primary", disabled=disabled):
        if not _check_api_keys(provider_label):
            st.error("Fehlender API-Key. Setze OPENAI_API_KEY (OpenAI) bzw. GOOGLE_API_KEY/GEMINI_API_KEY (Gemini).")
        else:
            script = _script_path(provider)
            if not script.exists():
                st.error(f"Skript nicht gefunden: {script}")
            else:
                PROGRESS_PATH.write_text(json.dumps({
                    "provider": provider, "processed": 0, "saved_ok": 0,
                    "target": int(limit_target), "status": "starting"
                }), encoding="utf-8")
                try:
                    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(LOG_PATH, "w", encoding="utf-8") as f:
                        f.write("=== START ===\n")
                except Exception:
                    pass

                args = [sys.executable, str(script),
                        "--campaign-slug", campaign_slug.strip(),
                        "--limit", str(int(limit_target)),
                        "--fetch", str(int(fetch_n)),
                        "--model", model,
                        "--ad-ids-file", str(ADIDS_TODAY_PATH),
                        "--progress-file", str(PROGRESS_PATH)]

                # Flags aus UI
                if analyze_all:
                    args += ["--analyze-all"]

                # Prompt-Override, falls vorhanden
                if use_custom_prompt:
                    PROMPT_OVERRIDE_PATH.write_text(prompt_text or DEFAULT_PROMPT, encoding="utf-8")
                    args += ["--prompt-file", str(PROMPT_OVERRIDE_PATH)]

                # Subprozess-Umgebung mit Rate-Limit-Parametern
                child_env = os.environ.copy()
                child_env["ANALYZE_ALL"] = "1" if analyze_all else "0"
                child_env["RL_TPM_HEADROOM"] = str(int(rl_tpm_headroom))
                child_env["RL_RPM_HEADROOM"] = str(int(rl_rpm_headroom))
                child_env["RL_MAX_CONCURRENCY"] = str(int(rl_conc))
                child_env["RL_BASE_SLEEP_MS"] = str(int(rl_sleep))
                child_env["RL_MAX_RETRIES"] = str(int(rl_retries))
                child_env["RL_JITTER_MS"] = str(int(rl_jitter))

                import subprocess
                creationflags = 0x08000000 if os.name == "nt" else 0
                try:
                    logf = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
                    proc = subprocess.Popen(
                        args, stdout=logf, stderr=logf,
                        creationflags=creationflags, env=child_env
                    )
                    st.session_state["llm_pid"] = proc.pid
                    st.session_state["llm_running"] = True
                    st.session_state["did_done_reload"] = False
                    st.success(f"Analyse gestartet. (PID {proc.pid})")
                except Exception as e:
                    st.error(f"Start fehlgeschlagen: {e}")

with c2:
    disabled = st.session_state["llm_pid"] is None
    if st.button("⏹️ Lauf abbrechen", disabled=disabled):
        pid = st.session_state["llm_pid"]
        if pid:
            try:
                import subprocess as _sp, os as _os, signal
                if os.name == "nt":
                    _sp.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                else:
                    _os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        st.session_state["llm_pid"] = None
        st.session_state["llm_running"] = False
        st.info("Prozess gestoppt.")

# ---------- Fortschritt ----------
status_box = st.empty()
try:
    data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    saved_ok   = int(data.get("saved_ok", 0))
    target     = int(data.get("target", 0))
    processed  = int(data.get("processed", 0))
    status     = data.get("status") or ""
    provider_p = (data.get("provider") or "?").upper()

    skipped_existing  = int(data.get("skipped_existing", 0))
    merged_ok         = int(data.get("merged_ok", 0))
    #unsuitable_local  = int(data.get("unsuitable_local", 0))
    no_b64            = int(data.get("no_b64", 0))
    openai_err        = int(data.get("openai_err", 0))
    #unsuitable_llm    = int(data.get("unsuitable_llm", 0))
    db_save_err       = int(data.get("db_save_err", 0))

    status_box.info(
        f"{provider_p} – gespeichert: {saved_ok}/{target} • verarbeitet: {processed} • "
        f"status={status} • skipped={skipped_existing} • merged={merged_ok} • "
        #f"ungeeignet(local)={unsuitable_local} • ungeeignet(llm)={unsuitable_llm} • "
        f"no_b64={no_b64} • openai_err={openai_err} • db_err={db_save_err}"
    )

    if target > 0:
        st.progress(min(1.0, saved_ok / max(1, target)), text=f"{saved_ok} von {target} Analysen gespeichert")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Skipped (bereits analysiert)", skipped_existing)
    #m2.metric("Ungeeignet (lokal)", unsuitable_local)
    #m3.metric("Ungeeignet (LLM)", unsuitable_llm)
    m4.metric("Kein Base64", no_b64)
    m5.metric("API-Errors", openai_err)
    m6.metric("DB-Speicherfehler", db_save_err)
    m7.metric("Fused OK", merged_ok)

    if status == "done" and not st.session_state.get("did_done_reload", False):
        st.session_state["llm_running"] = False
        st.session_state["did_done_reload"] = True
        time.sleep(0.2)
        st.rerun()
except Exception:
    pass

st.divider()

# ---------- Ergebnisse (JSON + Screenshot) ----------
st.subheader("Gespeicherte Ergebnisse ansehen")

df = pd.DataFrame()
if conn and campaign_slug.strip():
    df = _fetch_results(conn, campaign_slug.strip(), provider_ui=provider_label, limit=500)

if df.empty:
    st.info("Noch keine Analysen gespeichert.")
else:
    def _fmt_label(r):
        page = (r.get("page_name") or "").strip() or "Unbekannte Gruppe"
        ext  = r.get("ad_external_id") or "-"
        # Wunsch: nur "Gruppe (external_id)" – kein Punkt, kein Datum
        return f"{page} ({ext})"

    df["label"] = df.apply(_fmt_label, axis=1)
    opts = df["label"].tolist()
    sel  = st.selectbox("Ergebnis auswählen", options=opts, index=0)
    row  = df.iloc[opts.index(sel)]

    raw_payload = row.get("json_result", None)
    if raw_payload is None and "result" in row:
        raw_payload = row["result"]

    try:
        res_json = raw_payload if isinstance(raw_payload, (dict, list)) else json.loads(raw_payload)
    except Exception:
        res_json = raw_payload

    colL, colR = st.columns([1,1])
    b64, mime, cap = _fetch_media_b64_for_result(conn, res_json if isinstance(res_json, dict) else {}, int(row["ad_id"]))

    with colL:
        if b64 and mime:
            st.image(f"data:{mime};base64,{b64}", caption=cap, use_container_width=True)
        else:
            st.info("Kein Screenshot verfügbar.")

    with colR:
        try:
            st.json(res_json)
        except Exception:
            st.code(str(res_json), language="json")

st.divider()

# ---------- Logs ----------
with st.expander(f"Logs ({LOG_PATH})", expanded=False):
    try:
        txt = LOG_PATH.read_text(encoding="utf-8")
        MAX_CHARS = 20000
        if len(txt) > MAX_CHARS:
            txt = "…(gekürzt)…\n" + txt[-MAX_CHARS:]
        st.code(txt or "(leer)", language="bash")
    except Exception as e:
        st.info(f"(Noch kein Log vorhanden) {e}")

# ---------- Auto-Refresh nur während Run, am ENDE der Seite ----------
if st.session_state["llm_running"]:
    time.sleep(3)
    st.rerun()
