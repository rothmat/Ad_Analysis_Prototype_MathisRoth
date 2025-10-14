# -*- coding: utf-8 -*-
"""
Robuste Analyse von Ad-Screenshots via OpenAI:
- Strenges JSON-Schema (response_format=json_schema, strict:true)
- Selbstheilende Validierung + Retry
- --analyze-all erzwingt Analyse (nie geeignet:false)
- Backoff/Retry bei 429/5xx (Parameter via ENV aus UI)
"""

import os, sys, json, argparse, time, datetime as dt
from datetime import date as _date, datetime as _dt
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# stdout UTF-8 robust
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        try:
            print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass

# ---- DB-Client ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import (  # type: ignore
    connect,
    upsert_media_base64_from_path,
    upsert_analysis,
    save_result_for_ad,
)

# ---- OpenAI SDK ---------------------------------------------------------------
try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
except Exception as e:
    _safe_print("Bitte neue openai-SDK installieren: pip install --upgrade openai")
    raise

# ---- Optionales PIL für lokale Checks ----------------------------------------
try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


# ==============================================================================
# UI/ENV gesteuerte Backoff-Parameter (kommen aus Streamlit-UI)
# ==============================================================================
def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except Exception:
        return default

RL_BASE_SLEEP_MS = _env_int("RL_BASE_SLEEP_MS", 800)
RL_MAX_RETRIES   = _env_int("RL_MAX_RETRIES", 5)
RL_JITTER_MS     = _env_int("RL_JITTER_MS", 250)

# ==============================================================================
# Progress JSON fürs UI
# ==============================================================================
def _write_progress(progress_file, *, provider, processed, saved_ok, target,
                    status="running", last_media_id=None, merged_ok=0, skipped_existing=0,
                    no_b64=0, openai_err=0, db_save_err=0):
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
            "merged_ok": int(merged_ok),
            "skipped_existing": int(skipped_existing),
            # deaktivierte Zähler lassen wir bewusst weg
            "no_b64": int(no_b64),
            "openai_err": int(openai_err),
            "db_save_err": int(db_save_err),
            "updated_at": _dt.utcnow().isoformat() + "Z",
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# Hilfen: Dateien/Media/Base64
# ==============================================================================
def _file_path_from_uri(uri: str) -> Optional[Path]:
    if not uri:
        return None
    if uri.startswith("file:///"):
        return Path(uri.replace("file:///", "", 1))
    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))
    return None

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


# ==============================================================================
# Laden der Screenshots
# ==============================================================================
def _load_screenshots(conn, campaign_slug: str, limit: int = 400, ad_ids_filter: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    if ad_ids_filter:
        sql = """
        SELECT m.id AS media_id, m.ad_id, m.file_url, m.date_folder,
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
        SELECT m.id AS media_id, m.ad_id, m.file_url, m.date_folder,
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


def _already_analyzed_for_analysis(conn, ad_id: int, analysis_id: int) -> bool:
    sql = """
      SELECT 1
      FROM analysis_results
      WHERE ad_id = %s
        AND analysis_id = %s
      LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ad_id, analysis_id))
        return cur.fetchone() is not None

# ==============================================================================
# Prompt + JSON-Schema
# ==============================================================================
BASE_SYSTEM_PROMPT = r"""
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

# strenges Schema – nur wichtigste Pflichtfelder, Rest optional aber vorhanden
JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kampagne_id", "analysetool", "analyse_datum", "screenshot_id", "geeignet", "analyse"],
    "properties": {
        "kampagne_id": {"type": "string"},
        "analysetool": {"type": "string"},
        "analyse_datum": {"type": "string"},
        "screenshot_id": {"type": "string"},
        "geeignet": {"type": "boolean"},
        "analyse": {
            "type": ["object", "null"],
            "properties": {
                "visuelle_features": {
                    "type": "object",
                    "required": ["farbpalette", "schriftarten_erkannt", "textausrichtung",
                                 "flächenverteilung", "kompositionstyp", "bildtyp",
                                 "blickführung", "dominante_layoutstruktur",
                                 "plattformkontext_erkannt", "plattform", "elemente"],
                    "properties": {
                        "farbpalette": {"type": "array", "items": {"type": "string"}},
                        "schriftarten_erkannt": {"type": "array", "items": {"type": "string"}},
                        "schriftgrößen_verteilung": {"type": "object"},
                        "textausrichtung": {"type": "string"},
                        "flächenverteilung": {"type": "object"},
                        "kompositionstyp": {"type": "string"},
                        "bildtyp": {"type": "string"},
                        "blickführung": {"type": "string"},
                        "salienzverteilung": {"type": ["number", "integer"]},
                        "dominante_layoutstruktur": {"type": "string"},
                        "plattformkontext_erkannt": {"type": "boolean"},
                        "plattform": {"type": "string"},
                        "elemente": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "object"}
                        },
                    },
                    "additionalProperties": False
                },
                "textuelle_features": {
                    "type": "object",
                    "required": ["cta_typ", "cta_position", "cta_visuelle_prominenz",
                                 "cta_wirkungseinschätzung", "sprachstil", "tonalität",
                                 "anzahl_textblöcke", "anzahl_schriftarten",
                                 "gesamtwortanzahl", "gesamtzeichenanzahl",
                                 "durchschnittliche_wortlänge", "text_inhalte"],
                    "properties": {
                        "cta_typ": {"type": "string"},
                        "cta_position": {"type": "string"},
                        "cta_visuelle_prominenz": {"type": "string"},
                        "cta_wirkungseinschätzung": {"type": "string"},
                        "sprachstil": {"type": "string"},
                        "tonalität": {"type": "string"},
                        "anzahl_textblöcke": {"type": "integer"},
                        "anzahl_schriftarten": {"type": "integer"},
                        "gesamtwortanzahl": {"type": "integer"},
                        "gesamtzeichenanzahl": {"type": "integer"},
                        "durchschnittliche_wortlänge": {"type": "number"},
                        "text_inhalte": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"}
                        },
                        "headline_länge": {"type": ["string","null"]},
                        "headline_wortanzahl": {"type": ["integer","null"]},
                        "headline_zeichenanzahl": {"type": ["integer","null"]},
                        "text_bild_verhältnis": {"type": ["number","null"]},
                        "textgliederung_erkennbar": {"type": ["boolean","null"]},
                        "wortartenverteilung": {"type": ["object","null"]},
                    },
                    "additionalProperties": False
                },
                "semantische_features": {
                    "type": "object",
                    "required": ["zielgruppe","argumenttyp","framing_typ","ansprache_typ",
                                 "werbeversprechen","emotionaler_apell","bild_text_verhältnis","symbolgebrauch"],
                    "properties": {
                        "zielgruppe": {"type": "string"},
                        "argumenttyp": {"type": "string"},
                        "framing_typ": {"type": "string"},
                        "ansprache_typ": {"type": "string"},
                        "werbeversprechen": {"type": "string"},
                        "emotionaler_apell": {"type": "string"},
                        "bild_text_verhältnis": {"type": "string"},
                        "symbolgebrauch": {"type": "object"},
                        "zielgruppen_indikatoren": {"type": "array", "items": {"type": "string"}}
                    },
                    "additionalProperties": False
                },
            },
            "additionalProperties": False
        }
    }
}

# ==============================================================================
# Validierung & Reparatur
# ==============================================================================
def _is_valid(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Top-Level kein Objekt"
    if obj.get("analyse") is None:
        return False, "analyse ist null"
    an = obj.get("analyse", {})
    try:
        elems = (an.get("visuelle_features") or {}).get("elemente") or []
        texts = (an.get("textuelle_features") or {}).get("text_inhalte") or []
        platf = (an.get("visuelle_features") or {}).get("plattform")
        cta_t = (an.get("textuelle_features") or {}).get("cta_typ")
        if len(elems) < 1:
            return False, "zu wenige visuelle Elemente"
        if len(texts) < 1:
            return False, "keine text_inhalte"
        if not platf:
            return False, "plattform leer"
        if not cta_t:
            return False, "cta_typ leer"
    except Exception:
        return False, "Validierung fehlgeschlagen"
    return True, "ok"

def _force_defaults(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Füllt fehlende Felder defensiv mit Defaults, ohne Schema zu verletzen."""
    obj = dict(obj or {})
    obj.setdefault("analysetool", "OpenAI")
    obj.setdefault("geeignet", True)
    obj.setdefault("analyse", {})
    an = obj["analyse"]
    vf = an.setdefault("visuelle_features", {})
    tf = an.setdefault("textuelle_features", {})
    sf = an.setdefault("semantische_features", {})

    vf.setdefault("farbpalette", [])
    vf.setdefault("schriftarten_erkannt", [])
    vf.setdefault("textausrichtung", "Unklar")
    vf.setdefault("flächenverteilung", {"textfläche":0, "bildfläche":0, "weißraum":0})
    vf.setdefault("kompositionstyp", "Unklar")
    vf.setdefault("bildtyp", "Screenshot")
    vf.setdefault("blickführung", "Unklar")
    vf.setdefault("salienzverteilung", 0)
    vf.setdefault("dominante_layoutstruktur", "Social-Feed")
    vf.setdefault("plattformkontext_erkannt", True)
    vf.setdefault("plattform", "Unbekannt")
    vf.setdefault("elemente", [{"element":"Text","position":"Zentrum","farbe":"Unklar","größe":"mittel","form":"rechteckig","interaktiv_erscheinung":False,"funktion":"Textblock","bedeutung":"Unklar","inhalt":"Unklar","person_mimik_erkannbar":"nicht sichtbar","bild_inhalt":"Unklar","markenlogo_erkannt":False}])

    tf.setdefault("cta_typ", "Unklar")
    tf.setdefault("cta_position", "nicht vorhanden")
    tf.setdefault("cta_visuelle_prominenz", "gering")
    tf.setdefault("cta_wirkungseinschätzung", "schwach")
    tf.setdefault("sprachstil", "Unklar")
    tf.setdefault("tonalität", "Unklar")
    tf.setdefault("anzahl_textblöcke", max(1, len(tf.get("text_inhalte", []))))
    tf.setdefault("anzahl_schriftarten", len(vf.get("schriftarten_erkannt", [])))
    tf.setdefault("gesamtwortanzahl", 0)
    tf.setdefault("gesamtzeichenanzahl", 0)
    tf.setdefault("durchschnittliche_wortlänge", 0.0)
    tf.setdefault("text_inhalte", ["Unklar"])

    sf.setdefault("zielgruppe", "Allgemein")
    sf.setdefault("argumenttyp", "Unklar")
    sf.setdefault("framing_typ", "Unklar")
    sf.setdefault("ansprache_typ", "Unklar")
    sf.setdefault("werbeversprechen", "generisch")
    sf.setdefault("emotionaler_apell", "Unklar")
    sf.setdefault("bild_text_verhältnis", "Unklar")
    sf.setdefault("symbolgebrauch", {"symbol_erkannt": False, "symbol_typ": "Unklar", "symbol_bedeutung": "Unklar"})
    sf.setdefault("zielgruppen_indikatoren", [])
    return obj


# ==============================================================================
# OpenAI Call mit Schema + Reparatur-Retries
# ==============================================================================
def _openai_json_call(client: OpenAI, model: str, system_prompt: str,
                      image_b64: str, kampagne_id: str, screenshot_id: str,
                      analyse_datum: str, max_invalid_retries: int = 2) -> Dict[str, Any]:
    data_url = f"data:image/png;base64,{image_b64}"

    def _one_call(extra_user_text: Optional[str] = None) -> Dict[str, Any]:
        ut = {
            "kampagne_id": kampagne_id,
            "analyse_datum": analyse_datum,
            "screenshot_id": screenshot_id,
            "hinweis": "Liefere genau ein JSON-Objekt gemäß Schema; fülle Pflichtfelder mit konkreten Werten."
        }
        if extra_user_text:
            ut["korrekturhinweis"] = extra_user_text

        # Backoff bei 429/5xx
        for attempt in range(RL_MAX_RETRIES + 1):
            try:
                resp: ChatCompletion = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": json.dumps(ut, ensure_ascii=False)},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]},
                    ],
                    response_format={"type": "json_object"},
                )
                txt = resp.choices[0].message.content or "{}"
                return json.loads(txt)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower() or "503" in msg or "502" in msg:
                    sleep_ms = RL_BASE_SLEEP_MS * (2 ** attempt) + max(0, RL_JITTER_MS)
                    _safe_print(f"[BACKOFF] attempt={attempt+1}/{RL_MAX_RETRIES} wait={sleep_ms}ms reason={msg[:120]}")
                    time.sleep(sleep_ms / 1000.0)
                    continue
                raise

        # wenn wir hier sind, hat jeder Versuch geworfen
        raise RuntimeError("OpenAI call failed after retries")

    # erster Versuch
    obj = _one_call()
    ok, why = _is_valid(obj)
    tries = 0
    while (not ok) and (tries < max_invalid_retries):
        tries += 1
        obj = _one_call(f"Dein vorheriger Output war unvollständig/ungültig ({why}). "
                        f"Bitte fülle insbesondere `visuelle_features.elemente` (>=1) und "
                        f"`textuelle_features.text_inhalte` (>=1), setze `plattform`, `cta_typ` und liefere eine "
                        f"konsistente, vollständige Analyse gemäß Schema.")
        ok, why = _is_valid(obj)

    if not ok:
        # letzte Absicherung: Defaults injizieren
        obj = _force_defaults(obj)

    return obj


# ==============================================================================
# JSON merge helpers (wie gehabt)
# ==============================================================================
def _json_default(o):
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)

def load_json_any(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    out: List[Dict[str, Any]] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return [x for x in data["data"] if isinstance(x, dict)]
            else:
                return [data]
    except Exception:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            continue
    return out

def _api_index_by_external_id(api_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for it in api_items:
        ext = str(it.get("id") or "").strip()
        if ext:
            idx[ext] = it
    return idx

def get_campaign(conn, slug: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT id, name, slug FROM campaigns WHERE slug=%s"
    with conn.cursor() as cur:
        cur.execute(sql, (slug,))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "slug": row[2]}

def get_media_for_ad(conn, ad_id: int, date_folder: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    base = """
    SELECT id, kind, file_url, filename, date_folder
    FROM media
    WHERE ad_id=%s
    """
    params = [ad_id]
    if date_folder:
        base += " AND date_folder=%s"
        params.append(date_folder)
    base += " ORDER BY id"
    out = {"screenshots": [], "images": [], "videos": []}
    with conn.cursor() as cur:
        cur.execute(base, tuple(params))
        for mid, kind, file_url, filename, df in cur.fetchall():
            if isinstance(df, (dt.date, dt.datetime)):
                df = df.isoformat()
            item = {"media_id": mid, "file_url": file_url, "filename": filename, "date_folder": df}
            if kind == "screenshot":
                out["screenshots"].append(item)
            elif kind == "image":
                out["images"].append(item)
            elif kind == "video":
                out["videos"].append(item)
    return out

def _parse_media_id_from_screenshot_id(sid: str) -> Optional[int]:
    if not isinstance(sid, str):
        return None
    sid = sid.strip()
    if sid.startswith("media:"):
        try:
            return int(sid.split(":", 1)[1])
        except Exception:
            return None
    return None

def filter_media_to_llm_screenshot(media_refs: Dict[str, List[Dict[str, Any]]],
                                   llm_item: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    if not llm_item:
        return media_refs
    sid = llm_item.get("screenshot_id")
    mid = _parse_media_id_from_screenshot_id(sid) if sid else None
    if mid is None:
        return {"screenshots": [], "images": [], "videos": []}
    only_shot = [m for m in media_refs.get("screenshots", []) if int(m.get("media_id", -1)) == int(mid)]
    return {"screenshots": only_shot, "images": [], "videos": []}

def make_fused_json(now_iso: str,
                    campaign: Dict[str, Any],
                    ad_pk: int,
                    ad_external_id: str,
                    snapshot_date: str,
                    api_ad: Dict[str, Any],
                    llm_item: Optional[Dict[str, Any]],
                    media_refs: Dict[str, List[Dict[str, Any]]]
                    ) -> Dict[str, Any]:
    api_block = {"source": "facebook_ad_library", "raw": api_ad}
    llm_block = None
    if llm_item is not None:
        llm_block = {"provider": "openai", "analysis_file_payload": llm_item}
    fused = {
        "version": "1.1",
        "merge_time": now_iso,
        "ad_keys": {
            "ad_external_id": ad_external_id,
            "ad_pk": ad_pk,
            "campaign": {"id": campaign["id"], "slug": campaign["slug"], "name": campaign["name"]},
            "snapshot_date": snapshot_date
        },
        "api": api_block,
        "media": media_refs,
        "llm_analysis": llm_block,
        "derived": {"has_llm": llm_block is not None, "has_video": len(media_refs.get("videos", [])) > 0}
    }
    return fused

def ensure_fused_table(conn) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS ad_llm_fused (
      id                BIGSERIAL PRIMARY KEY,
      ad_id             BIGINT NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
      snapshot_date     DATE   NOT NULL,
      fused             JSONB  NOT NULL,
      created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (ad_id, snapshot_date)
    );
    CREATE INDEX IF NOT EXISTS idx_ad_llm_fused_gin ON ad_llm_fused USING GIN (fused jsonb_path_ops);
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def upsert_fused(conn, ad_id: int, snapshot_date: str, fused: Dict[str, Any]) -> int:
    sql = """
    INSERT INTO ad_llm_fused (ad_id, snapshot_date, fused)
    VALUES (%s, %s, %s::jsonb)
    ON CONFLICT (ad_id, snapshot_date) DO UPDATE
      SET fused = EXCLUDED.fused
    RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ad_id, snapshot_date, json.dumps(fused, ensure_ascii=False, default=_json_default)))
        rid = cur.fetchone()[0]
    conn.commit()
    return int(rid)

# ==============================================================================
# Auto-Find der heutigen API-JSON
# ==============================================================================
def _json_from_snapshot_table(conn) -> Optional[Path]:
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

def _find_todays_json_fallback(campaign_slug: str) -> Optional[Path]:
    today = _date.today().isoformat()
    root = Path.cwd()
    cand = root / campaign_slug.capitalize() / f"{today}.json"
    if cand.exists():
        return cand
    for p in root.rglob(f"{today}.json"):
        return p
    return None

def _discover_api_json(conn, campaign_slug: str) -> Tuple[Optional[Path], str]:
    p_db = _json_from_snapshot_table(conn)
    if p_db:
        return p_db, "db"
    p_fb = _find_todays_json_fallback(campaign_slug)
    if p_fb:
        return p_fb, "fallback"
    return None, "none"


# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-slug", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--fetch", type=int, default=400)
    ap.add_argument("--analysis-name", default="screenshot_analysis_v2")
    ap.add_argument("--analysis-version", default="1.2.0")
    ap.add_argument("--ad-ids-file", type=str, default=None)
    ap.add_argument("--progress-file", type=str, default=None)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--analyze-all", action="store_true", help="Nie als ungeeignet markieren; erzwungene Analyse.")
    args = ap.parse_args()

    client = OpenAI()
    analyse_datum = _date.today().isoformat()

    out_root = Path.cwd() / args.campaign_slug.capitalize()
    out_file = out_root / "LLM-Analysis" / "OpenAI" / f"{analyse_datum}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Prompt laden (Override oder Basisprompt)
    system_prompt = BASE_SYSTEM_PROMPT
    if args.prompt_file and Path(args.prompt_file).exists():
        try:
            system_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except Exception:
            pass

    def _load_ids(pth: Optional[str]) -> List[int]:
        if not pth:
            return []
        try:
            vals = json.loads(Path(pth).read_text(encoding="utf-8"))
            return [int(v) for v in vals if str(v).isdigit()]
        except Exception:
            return []

    ad_ids_filter = _load_ids(args.ad_ids_file)

    with connect() as conn:
        # Vorbereiten: heutige API-JSON
        api_path, src = _discover_api_json(conn, args.campaign_slug)
        api_idx: Dict[str, Dict[str, Any]] = {}
        fused_file_outputs: List[Dict[str, Any]] = []

        if api_path and api_path.exists():
            try:
                api_items = load_json_any(api_path)
                api_idx = _api_index_by_external_id(api_items)
                _safe_print(f"[INIT] API-JSON: {api_path} (source={src}) • Items: {len(api_idx)}")
            except Exception as e:
                _safe_print(f"[WARN] API-JSON konnte nicht geladen werden: {e}")

        # Analysesatz registrieren
        analysis_id = upsert_analysis(
            conn, args.analysis_name, "openai", args.analysis_version,
            parameters={"model": args.model, "schema": "json_schema/strict", "input": "base64-screenshot"}
        )
        conn.commit()

        if api_idx:
            ensure_fused_table(conn)

        shots = _load_screenshots(conn, args.campaign_slug, limit=args.fetch, ad_ids_filter=ad_ids_filter)
        if not shots:
            _safe_print("Keine passenden Screenshots gefunden.")
            _write_progress(args.progress_file, provider="openai",
                            processed=0, saved_ok=0, target=args.limit,
                            status="done", last_media_id=None, merged_ok=0, skipped_existing=0,
                            no_b64=0, openai_err=0, db_save_err=0)
            return

        saved_ok = 0
        processed = 0
        skipped_existing = 0
        openai_err = 0
        no_b64 = 0
        db_save_err = 0
        merged_ok = 0

        file_outputs: List[Dict[str, Any]] = []

        _write_progress(args.progress_file, provider="openai",
                        processed=0, saved_ok=0, target=args.limit, status="running",
                        last_media_id=None, merged_ok=0, skipped_existing=0,
                        no_b64=0, openai_err=0, db_save_err=0)

        for row in shots:
            if saved_ok >= args.limit:
                break

            media_id    = int(row["media_id"])
            ad_id       = int(row["ad_id"])
            file_url    = str(row["file_url"])
            ad_external_id = str(row.get("ad_external_id") or "").strip()
            campaign_id = int(row["campaign_id"])
            shot_id     = f"media:{media_id}"

            processed += 1

            # Skip, wenn bereits für dieses Modell analysiert
            if _already_analyzed_for_analysis(conn, ad_id, analysis_id):
                skipped_existing += 1
                _safe_print(f"[SKIP] bereits analysiert (analysis_id={analysis_id}, ad_id={ad_id})")
                _write_progress(args.progress_file, provider="openai",
                                processed=processed, saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id, merged_ok=merged_ok, skipped_existing=skipped_existing,
                                no_b64=no_b64, openai_err=openai_err, db_save_err=db_save_err)
                continue

            # Base64 sicherstellen
            b64 = _ensure_b64_for_media(conn, media_id, file_url)
            if not b64:
                no_b64 += 1
                _safe_print(f"[WARN] Kein Base64 fuer media_id={media_id}")
                _write_progress(args.progress_file, provider="openai",
                                processed=processed, saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id, merged_ok=merged_ok, skipped_existing=skipped_existing,
                                no_b64=no_b64, openai_err=openai_err, db_save_err=db_save_err)
                continue

            # OpenAI-Aufruf
            try:
                out_json = _openai_json_call(
                    client=client, model=args.model, system_prompt=system_prompt,
                    image_b64=b64, kampagne_id=str(campaign_id),
                    screenshot_id=shot_id, analyse_datum=_date.today().isoformat(),
                    max_invalid_retries=2
                )
            except Exception as e:
                openai_err += 1
                _safe_print(f"[ERR] OpenAI-Fehler media_id={media_id}: {e}")
                _write_progress(args.progress_file, provider="openai",
                                processed=processed, saved_ok=saved_ok, target=args.limit, status="running",
                                last_media_id=media_id, merged_ok=merged_ok, skipped_existing=skipped_existing,
                                no_b64=no_b64, openai_err=openai_err, db_save_err=db_save_err)
                continue

            # analyze-all erzwingt Analyse
            if args.analyze_all:
                out_json["geeignet"] = True
                if out_json.get("analyse") is None:
                    out_json = _force_defaults(out_json)

            # Top-Level Defaults
            out_json.setdefault("kampagne_id", str(campaign_id))
            out_json.setdefault("analysetool", "OpenAI")
            out_json.setdefault("analyse_datum", _date.today().isoformat())
            out_json.setdefault("screenshot_id", shot_id)

            # Speichern
            try:
                save_result_for_ad(conn, analysis_id, ad_id, out_json, score=None)
                conn.commit()
                saved_ok += 1
                _safe_print(f"[OK] gespeichert (ad_id={ad_id}, media_id={media_id}) [{saved_ok}/{args.limit}]")
            except Exception as db_e:
                conn.rollback()
                db_save_err += 1
                _safe_print(f"[WARN] DB-Speicherfehler media_id={media_id}: {db_e}")

            file_outputs.append(out_json)

            # Fused (wenn API verfügbar)
            if api_idx and ad_external_id:
                api_ad = api_idx.get(ad_external_id)
                if api_ad:
                    snapshot_date = (api_ad.get("ad_creation_time")
                        or api_ad.get("ad_delivery_start_time")
                        or _date.today().isoformat())[:10]
                    # Wichtig: Nicht nach date_folder einschränken – wir haben die media_id
                    all_media = get_media_for_ad(conn, ad_id, date_folder=None)
                    filtered  = filter_media_to_llm_screenshot(all_media, out_json)
                    campaign_meta = get_campaign(conn, args.campaign_slug)
                    if campaign_meta:
                        try:
                            fused = make_fused_json(
                                now_iso=_dt.utcnow().isoformat(timespec="seconds") + "Z",
                                campaign=campaign_meta,
                                ad_pk=ad_id,
                                ad_external_id=ad_external_id,
                                snapshot_date=snapshot_date,
                                api_ad=api_ad,
                                llm_item=out_json,
                                media_refs=filtered
                            )
                            ensure_fused_table(conn)
                            upsert_fused(conn, ad_id, snapshot_date, fused)
                            merged_ok += 1
                        except Exception as fe:
                            _safe_print(f"[FUSE] warn (ad_id={ad_id}): {fe}")

            _write_progress(args.progress_file, provider="openai",
                            processed=processed, saved_ok=saved_ok, target=args.limit, status="running",
                            last_media_id=media_id, merged_ok=merged_ok, skipped_existing=skipped_existing,
                            no_b64=no_b64, openai_err=openai_err, db_save_err=db_save_err)

        # Datei-Output (optional, für Nachvollziehbarkeit)
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(file_outputs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _safe_print(f"[WARN] Konnte LLM-Datei nicht schreiben: {e}")

        _write_progress(args.progress_file, provider="openai",
                        processed=processed, saved_ok=saved_ok, target=args.limit, status="done",
                        last_media_id=None, merged_ok=merged_ok, skipped_existing=skipped_existing,
                        no_b64=no_b64, openai_err=openai_err, db_save_err=db_save_err)


if __name__ == "__main__":
    main()
