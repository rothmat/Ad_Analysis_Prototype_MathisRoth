# -*- coding: utf-8 -*-
"""
Analyse von Ad-Screenshots (base64) via Google Gemini nach festem JSON-Schema.
- Liest Screenshots (media.kind='screenshot') für eine Kampagne aus Postgres
- Erstellt bei Bedarf Base64 aus lokalem Pfad und speichert in media_base64
- Schickt pro Bild 1 Request an Gemini mit dem System-Prompt
- Stoppt nach N geeigneten Analysen (geeignet=true) oder wenn keine weiteren vorhanden sind
- Speichert jeden Output als vollständiges JSON in analysis_results (target='ad', provider='gemini')
- Zusätzlich: schreibt alle JSON-Outputs in <Campaign>/LLM-Analysis/Gemini/YYYY-MM-DD.json

ENV:
  GOOGLE_API_KEY oder GEMINI_API_KEY  (z.B. AIza...)
  DATABASE_URL                        (z.B. postgresql://app:app@localhost:5432/appdb)

Run (PowerShell):
  $env:GOOGLE_API_KEY = "<dein-key>"
  $env:DATABASE_URL   = "postgresql://app:app@localhost:5432/appdb"
  py .\agents\LLM-Initial-Analysis\gemini_image_analysis.py --campaign-slug eigenmietwert --limit 50 --fetch 400 --model gemini-1.5-pro
"""

import os, sys, json, argparse, datetime as dt
from datetime import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, Any, List

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- stdout/stderr robust auf UTF-8 stellen (Windows-safe) ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _safe_print(msg: str) -> None:
    """Druckt notfalls ohne nicht-ASCII-Zeichen, um Encoding-Errors zu vermeiden."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode("ascii"))

# --- db_client verfügbar machen -------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import (  # type: ignore
    connect,
    upsert_media_base64_from_path,
    upsert_analysis,
    save_result_for_ad,
)

# --- Optionales Pillow/Numpy (graues Mittelband) --------------------------------
try:
    from PIL import Image
    import numpy as np
    _HAVE_IMG = True
except Exception:
    _HAVE_IMG = False

# --- Gemini SDK -----------------------------------------------------------------
try:
    import google.generativeai as genai
except ImportError:
    print("Bitte 'pip install google-generativeai' installieren.")
    sys.exit(1)


# ===============================================================================
# Gemeinsame Utilities (Progress & ad_id-Filter)
# ===============================================================================
def _write_progress(progress_file, *, provider, processed, saved_ok, target,
                    status="running", last_media_id=None):
    if not progress_file:
        return
    try:
        p = Path(progress_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider,
            "processed": int(processed),
            "saved_ok": int(saved_ok),
            "target": int(target),
            "last_media_id": last_media_id,
            "status": status,
            "updated_at": _dt.utcnow().isoformat() + "Z",
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _load_ad_ids_from_file(pth: str | None) -> list[int]:
    if not pth:
        return []
    try:
        vals = json.loads(Path(pth).read_text(encoding="utf-8"))
        return [int(v) for v in vals if isinstance(v, (int, str)) and str(v).strip().isdigit()]
    except Exception:
        return []

# ------------------------------------------------------------
# Prompt (identische Logik wie OpenAI, weniger konservativ)
# ------------------------------------------------------------
SYSTEM_PROMPT = r"""
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
"""

# ===============================================================================
# Lokaler Qualitätsfilter (graues Mittelband)
# ===============================================================================
from typing import Optional as _Optional
def _file_path_from_uri(uri: str) -> _Optional[Path]:
    if not uri:
        return None
    if uri.startswith("file:///"):
        return Path(uri.replace("file:///", "", 1))
    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))
    return None

def _has_grey_center_band(p: Path) -> bool:
    if not _HAVE_IMG:
        return False
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w < 200 or h < 120:
            return False
        y0, y1 = int(h * 0.40), int(h * 0.62)
        crop = im.crop((0, y0, w, y1))
        arr = np.asarray(crop).astype("float32") / 255.0
        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        maxc = np.max(arr, axis=2); minc = np.min(arr, axis=2)
        sat = (maxc - minc)
        mid_gray = (lum > 0.35) & (lum < 0.75)
        low_sat  = sat < 0.12
        frac_midgray = float(np.mean(mid_gray))
        frac_lowsat  = float(np.mean(low_sat))
        lum_std = float(np.std(lum))
        return (frac_midgray > 0.65 and frac_lowsat > 0.70 and lum_std < 0.08)
    except Exception:
        return False

def _is_locally_unsuitable(file_uri: str) -> bool:
    p = _file_path_from_uri(file_uri)
    if not p or not p.exists():
        return False
    return _has_grey_center_band(p)


# ===============================================================================
# DB I/O
# ===============================================================================
def _ensure_b64_for_media(conn, media_id: int, file_uri: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT b64 FROM media_base64 WHERE media_id=%s", (media_id,))
        row = cur.fetchone()
        if row and row[0]:
            return row[0]

    p = _file_path_from_uri(file_uri)
    if not p or not p.exists():
        return None

    try:
        upsert_media_base64_from_path(conn, media_id, str(p))
        conn.commit()
    except Exception:
        conn.rollback()
        return None

    with conn.cursor() as cur:
        cur.execute("SELECT b64 FROM media_base64 WHERE media_id=%s", (media_id,))
        row2 = cur.fetchone()
        return row2[0] if row2 and row2[0] else None


def _load_screenshots(conn, campaign_slug: str, limit: int = 200,
                      ad_ids_filter: list[int] | None = None) -> List[Dict[str, Any]]:
    """
    Holt Screenshots (neueste zuerst) inkl. ad_id, ad_external_id, file_url, media_id, campaign_id.
    Wenn ad_ids_filter gesetzt ist, werden ausschließlich diese ad_id berücksichtigt.
    """
    if ad_ids_filter:
        sql = """
          SELECT m.id AS media_id, m.ad_id, m.filename, m.file_url, m.date_folder,
                 a.ad_external_id, c.id AS campaign_id
          FROM media m
          JOIN ads a ON a.id = m.ad_id
          JOIN campaigns c ON c.id = a.campaign_id
          WHERE c.slug = %s AND m.kind = 'screenshot'
            AND m.ad_id = ANY(%s)
          ORDER BY m.created_at DESC
          LIMIT %s
        """
        params = (campaign_slug, ad_ids_filter, limit)
    else:
        sql = """
          SELECT m.id AS media_id, m.ad_id, m.filename, m.file_url, m.date_folder,
                 a.ad_external_id, c.id AS campaign_id
          FROM media m
          JOIN ads a ON a.id = m.ad_id
          JOIN campaigns c ON c.id = a.campaign_id
          WHERE c.slug = %s AND m.kind = 'screenshot'
          ORDER BY m.created_at DESC
          LIMIT %s
        """
        params = (campaign_slug, limit)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ===============================================================================
# Gemini Call
# ===============================================================================
def _call_gemini(model: "genai.GenerativeModel", b64_png: str,
                 kampagne_id: str, screenshot_id: str, analyse_datum: str) -> Dict[str, Any]:
    meta_text = json.dumps({
        "kampagne_id": kampagne_id,
        "analyse_datum": analyse_datum,
        "screenshot_id": screenshot_id,
        "hinweis": "Gib ausschließlich ein einzelnes JSON-Objekt nach dem vorgegebenen Schema aus."
    }, ensure_ascii=False)

    import base64 as _b64
    try:
        image_bytes = _b64.b64decode(b64_png)
    except Exception:
        raise RuntimeError("Base64 konnte nicht dekodiert werden.")

    resp = model.generate_content(
        contents=[SYSTEM_PROMPT, meta_text, {"mime_type": "image/png", "data": image_bytes}],
        generation_config={
            "temperature": 0,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
        }
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Leere Antwort vom Modell.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("Antwort ist kein valides JSON.")


# ===============================================================================
# Main
# ===============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-slug", required=True, help="z.B. eigenmietwert")
    ap.add_argument("--model", default="gemini-1.5-pro")
    ap.add_argument("--limit", type=int, default=10, help="Anzahl geeigneter Analysen (Ziel)")
    ap.add_argument("--fetch", type=int, default=100, help="wie viele Screenshots maximal aus DB holen")
    ap.add_argument("--analysis-name", default="screenshot_analysis_v1")
    ap.add_argument("--analysis-version", default="1.0.1")
    # ✨ neu:
    ap.add_argument("--ad-ids-file", type=str, default=None,
                    help="Pfad zu JSON-Liste interner ad_id (nur diese werden analysiert)")
    ap.add_argument("--progress-file", type=str, default=None,
                    help="Pfad zu einer Progress-JSON für das UI")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Fehlender GOOGLE_API_KEY/GEMINI_API_KEY in der Umgebung.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    analyse_datum = dt.date.today().isoformat()

    out_root = Path.cwd() / args.campaign_slug.capitalize()
    out_file = out_root / "LLM-Analysis" / "Gemini" / f"{analyse_datum}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dateiausgabe: {out_file}")

    # Filter-Liste laden + initialen Progress schreiben
    ad_ids_filter = _load_ad_ids_from_file(args.ad_ids_file)
    _write_progress(args.progress_file, provider="gemini", processed=0, saved_ok=0,
                    target=args.limit, status="starting")

    with connect() as conn:
        analysis_id = upsert_analysis(
            conn, args.analysis_name, "gemini", args.analysis_version,
            parameters={"model": args.model, "schema": "fixed-json", "input": "base64-screenshot"}
        )
        conn.commit()

        shots = _load_screenshots(conn, args.campaign_slug, limit=args.fetch, ad_ids_filter=ad_ids_filter)
        if not shots:
            print("Keine passenden Screenshots gefunden.")
            _write_progress(args.progress_file, provider="gemini", processed=0, saved_ok=0,
                            target=args.limit, status="done")
            return

        saved_ok = 0
        processed = 0
        file_outputs: List[Dict[str, Any]] = []

        for row in shots:
            if saved_ok >= args.limit:
                break
            processed += 1

            media_id    = row["media_id"]
            ad_id       = row["ad_id"]
            file_url    = row["file_url"]
            campaign_id = row["campaign_id"]
            shot_id     = f"media:{media_id}"

            # 1) lokaler Pre-Filter
            if _is_locally_unsuitable(file_url):
                file_outputs.append({
                    "kampagne_id": str(campaign_id),
                    "analysetool": "OpenAI",  # Schema verlangt wörtlich "OpenAI"
                    "analyse_datum": analyse_datum,
                    "screenshot_id": shot_id,
                    "geeignet": False,
                    "analyse": None
                })
                _write_progress(args.progress_file, provider="gemini", processed=processed,
                                saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id)
                continue

            # 2) Base64 holen/erzeugen
            b64 = _ensure_b64_for_media(conn, media_id, file_url)
            if not b64:
                _write_progress(args.progress_file, provider="gemini", processed=processed,
                                saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id)
                continue

            # 3) Gemini Aufruf
            try:
                out_json = _call_gemini(
                    model=model, b64_png=b64, kampagne_id=str(campaign_id),
                    screenshot_id=shot_id, analyse_datum=analyse_datum
                )
            except Exception as e:
                print(f"  ❌ Gemini-Fehler media_id={media_id}: {e}")
                _write_progress(args.progress_file, provider="gemini", processed=processed,
                                saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id)
                continue

            out_json.setdefault("kampagne_id", str(campaign_id))
            out_json.setdefault("analysetool", "OpenAI")  # Feldname im Schema
            out_json.setdefault("analyse_datum", analyse_datum)
            out_json.setdefault("screenshot_id", shot_id)
            if "geeignet" not in out_json:
                out_json["geeignet"] = True

            if out_json.get("geeignet") is False:
                file_outputs.append(out_json)
                _write_progress(args.progress_file, provider="gemini", processed=processed,
                                saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id)
                continue

            # 4) DB speichern
            try:
                save_result_for_ad(conn, analysis_id, ad_id, out_json, score=None)
                conn.commit()
                saved_ok += 1
                print(f"  ✓ gespeichert (ad_id={ad_id}, media_id={media_id})  [{saved_ok}/{args.limit}]")
            except Exception as db_e:
                conn.rollback()
                print(f"  ⚠️  DB-Speicherfehler media_id={media_id}: {db_e}")

            file_outputs.append(out_json)

            _write_progress(args.progress_file, provider="gemini", processed=processed,
                            saved_ok=saved_ok, target=args.limit, status="running",
                            last_media_id=media_id)

        # Datei-Output
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(file_outputs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠️  Konnte Datei nicht schreiben: {e}")

        _write_progress(args.progress_file, provider="gemini", processed=processed,
                        saved_ok=saved_ok, target=args.limit, status="done")


if __name__ == "__main__":
    main()