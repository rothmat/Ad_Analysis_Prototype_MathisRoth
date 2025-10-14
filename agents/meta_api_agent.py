# agents/meta_api_agent.py
"""
Meta Ad Library Tracker – parametrisiert & UI-ready
- Eingaben: Thema (Kampagne), Region (Country), Zeitraum
- Keyword-Generator (DE/FR/IT), standardmäßig 12 je Sprache
- Holt Ads von Meta Ad Library, dedupliziert, speichert optional in DB und schreibt Tagesdatei
- Kann aus Streamlit/CLI aufgerufen werden und liefert Fortschritt via Callback

ENV:
  META_ACCESS_TOKEN   = <dein Meta Graph Access Token>   (Pflicht)
  OPENAI_API_KEY      = <optional; für FR/IT-Basisterm-Übersetzung>
  DATABASE_URL        = postgresql://app:pw@127.0.0.1:5432/appdb (optional)
"""

import os
import json
import time
import datetime as dt
from typing import Dict, List, Optional, Callable, Any
import requests
from pathlib import Path
import sys

# --- Postgres Session-Guards (gegen Locks/Hänger) -----------------
PG_LOCK_TIMEOUT = os.getenv("PG_LOCK_TIMEOUT", "2s")
PG_IDLE_XACT_TIMEOUT = os.getenv("PG_IDLE_XACT_TIMEOUT", "2min")
PG_STMT_TIMEOUT = os.getenv("PG_STMT_TIMEOUT", "5min")

def _configure_pg_session(conn, app_name: str) -> None:
    """Setzt kurze Timeouts pro Session und vergibt application_name.
    Verhindert, dass DDL/UPSERTs minutenlang blockieren."""
    try:
        try:
            # optional: sicherstellen, dass wir nicht im Autocommit hängen
            conn.autocommit = False
        except Exception:
            pass

        cur = conn.cursor()
        # application_name hilft beim Debuggen in pg_stat_activity
        cur.execute("SET application_name = %s;", (app_name,))
        # kurze Lock/Idle/Statement-Timeouts
        cur.execute("SET lock_timeout = %s;", (PG_LOCK_TIMEOUT,))
        cur.execute("SET idle_in_transaction_session_timeout = %s;", (PG_IDLE_XACT_TIMEOUT,))
        cur.execute("SET statement_timeout = %s;", (PG_STMT_TIMEOUT,))
        cur.close()
        conn.commit()
    except Exception as e:
        # Fällt ohne zu blockieren zurück; wir wollen nie an dieser Stelle hängen bleiben
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"⚠️ Konnte Session-Settings nicht setzen: {e}")

# --- db_client laden (optional) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ad-db" / "ingest"))
try:
    from db_client import (  # type: ignore
        connect,
        get_or_create_campaign,
        save_api_snapshot,
        upsert_ad_with_raw,
    )
    _HAVE_DB = True
except Exception:
    _HAVE_DB = False

# --- OpenAI (optional) ---
try:
    from openai import OpenAI
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False

# Wenn leer (""), wird die Umgebungsvariable OPENAI_API_KEY verwendet.
OPENAI_API_KEY = ""  

BASE_URL = "https://graph.facebook.com/v23.0/ads_archive"
REQUEST_TIMEOUT = 40
RETRY_BACKOFFS = [2, 4, 8, 16]  # Sekunden

FIELDS = ",".join([
    "id","ad_creation_time","ad_delivery_start_time","ad_delivery_stop_time",
    "ad_creative_bodies","ad_creative_link_captions","ad_creative_link_descriptions",
    "ad_creative_link_titles","ad_snapshot_url","currency","demographic_distribution",
    "delivery_by_region","impressions","page_id","page_name","publisher_platforms",
    "spend","languages","bylines","estimated_audience_size","age_country_gender_reach_breakdown",
    "beneficiary_payers","eu_total_reach","target_ages","target_gender","target_locations",
    "total_reach_by_location",
])

ProgressFn = Optional[Callable[[str, float], None]]  # (message, progress 0..1)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _today_str(tz: Optional[dt.tzinfo] = None) -> str:
    return dt.datetime.now(tz).strftime("%Y-%m-%d")

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _filename_for_today(output_dir: str) -> str:
    _ensure_dir(output_dir)
    return os.path.join(output_dir, f"{_today_str()}.json")

def _get_with_retries(url: str, params: Dict[str, str]) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt, backoff in enumerate([0] + RETRY_BACKOFFS, start=1):
        if backoff:
            time.sleep(backoff)
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                continue
            return resp
        except requests.RequestException as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Unbekannter Fehler bei HTTP-Request")

def _fetch_one_query(base_params: Dict[str, str], query: str) -> List[Dict]:
    params = dict(base_params)
    params["search_terms"] = query

    all_rows: List[Dict] = []
    next_url: Optional[str] = None
    next_params: Optional[Dict[str, str]] = None

    while True:
        if next_url:
            resp = _get_with_retries(next_url, next_params or {})
        else:
            resp = _get_with_retries(BASE_URL, params)

        if resp.status_code != 200:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": f"HTTP {resp.status_code}", "text": resp.text[:500]}
            print(f"❌ Fehler für '{query}': {payload}")
            break

        payload = resp.json()
        data = payload.get("data", [])
        all_rows.extend(data)

        paging = payload.get("paging", {})
        cursors = paging.get("cursors", {})
        next_url = paging.get("next")
        if not next_url:
            break

        after = cursors.get("after")
        next_params = {"after": after} if after else {}
        time.sleep(0.2)

    return all_rows

def _dedupe_by_id(rows: List[Dict]) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    for r in rows:
        rid = str(r.get("id") or "")
        if rid:
            seen[rid] = r
    return list(seen.values())

def _slugify(name: str) -> str:
    return "-".join("".join(c for c in name.lower().strip() if c.isalnum() or c in " _-").split())

# ------------------------------------------------------------
# Keyword-Generator (deterministisch, recall-stark)
# ------------------------------------------------------------
def _openai_client() -> Optional[OpenAI]:
    if not _HAVE_OPENAI:
        return None
    key = OPENAI_API_KEY.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)

def _localized_bases(topic: str) -> Dict[str, Any]:
    """
    Liefert Hauptbegriff DE/FR/IT und – falls OpenAI verfügbar – je Sprache
    kurze, polit-werbungspezifische Phrasen (fr_variants / it_variants).
    """
    t = topic.strip()
    de = t

    # Heuristische Defaults
    low = t.lower()
    fr = t
    it = t
    if "eigenmietwert" in low:
        fr = "valeur locative"
        it = "valore locativo"

    fr_variants: List[str] = []
    it_variants: List[str] = []

    client = _openai_client()
    if client:
        try:
            prompt = (
                "Kontext: politische Werbung in der Schweiz (Meta Ad Library). "
                "Auf Basis des folgenden deutschen Policy-Themas sollst du die "
                "gebräuchliche fachliche Hauptbezeichnung auf Französisch und "
                "Italienisch bestimmen und je Sprache außerdem kurze, in "
                "Kampagnen häufig verwendete Suchphrasen liefern.\n"
                "Regeln:\n"
                "• Sachlich, neutral, keine Hashtags/Anführungszeichen.\n"
                "• Phrasen max. 3–5 Wörter, keine Marken/Parteien.\n"
                "• Fokus: Regulierung, Abschaffung/Reform, Steuer/Abgabe, Wohnen/Eigentum.\n"
                "Antworte als JSON mit genau diesen Schlüsseln:\n"
                "{\"fr\":\"...\",\"it\":\"...\",\"fr_variants\":[\"...\"],\"it_variants\":[\"...\"]}\n\n"
                f"Thema (DE): {t}"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")

            fr = (data.get("fr") or fr).strip()
            it = (data.get("it") or it).strip()

            def _norm_list(x):
                out = []
                for s in (x or []):
                    s = (s or "").strip()
                    if s and s not in out:
                        out.append(s)
                return out[:10]

            fr_variants = _norm_list(data.get("fr_variants"))
            it_variants = _norm_list(data.get("it_variants"))
        except Exception:
            pass

    return {"de": de, "fr": fr, "it": it, "fr_variants": fr_variants, "it_variants": it_variants}


def generate_keywords(
    topic: str,
    per_lang: int = 12,
    langs: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, List[str]]:
    """
    Erzeugt je Sprache exakt 'per_lang' Suchbegriffe.
    - DE: erstes Keyword ist IMMER das eingegebene Wort.
    - FR/IT: Hauptbegriff zuerst, dann LLM-Varianten, dann Templates.
    """
    alias = kwargs.get("per_language") or kwargs.get("n_per_lang") or kwargs.get("n")
    if alias is not None:
        per_lang = int(alias)

    per_lang = max(1, int(per_lang or 12))
    bases = _localized_bases(topic)
    t, tf, ti = bases["de"], bases["fr"], bases["it"]
    fr_vars: List[str] = bases.get("fr_variants", []) or []
    it_vars: List[str] = bases.get("it_variants", []) or []

    de_templates = [
        "Abschaffung {t}", "Besteuerung {t}", "{t} Steuer", "Wohneigentum {t}",
        "Systemwechsel {t}", "{t} abschaffen", "Steuer auf {t}",
        "Wohnsteuer Schweiz", "Hausbesitzer Steuer",
        "Wohneigentum fördern", "Systemwechsel Wohneigentum",
    ]
    fr_templates = [
        "suppression {tf}", "imposition {tf}", "{tf} logement", "système {tf}",
        "suppression de la {tf}", "imposition de la {tf}", "logement {tf}",
        "propriétaires immobiliers impôt", "réforme {tf}",
        "système fiscal logement", "impôt sur la {tf}",
    ]
    it_templates = [
        "abolizione {ti}", "tassazione {ti}", "{ti} abitazione", "sistema {ti}",
        "abolizione del {ti}", "tassazione del {ti}", "casa {ti}",
        "proprietari di casa imposta", "riforma {ti}",
        "sistema fiscale abitazione", "imposta sul {ti}",
    ]

    def _dedup_keep_order(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for s in seq:
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k); out.append(s.strip())
        return out

    # DE: Eingabewort zuerst
    de_list = _dedup_keep_order([t] + [tpl.format(t=t) for tpl in de_templates])[:per_lang]

    # FR/IT: Hauptbegriff → Varianten → Templates
    fr_list = _dedup_keep_order([tf] + fr_vars + [tpl.format(tf=tf) for tpl in fr_templates])[:per_lang]
    it_list = _dedup_keep_order([ti] + it_vars + [tpl.format(ti=ti) for tpl in it_templates])[:per_lang]

    # Speziell: „Eigenmietwert“ hart ersetzen, falls in Vorlagen auftaucht
    def _force_replace(lst: List[str], base: str) -> List[str]:
        repl = []
        for s in lst:
            repl.append(s.replace("Eigenmietwert", base).replace("eigenmietwert", base))
        return _dedup_keep_order(repl)

    fr_list = _force_replace(fr_list, tf)
    it_list = _force_replace(it_list, ti)

    return {"de": de_list, "fr": fr_list, "it": it_list}

# ------------------------------------------------------------
# Ergebnis-Aufbereitung fürs UI
# ------------------------------------------------------------
def _as_int(x):
    try:
        if x is None: return None
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None

def _mid(lb, ub):
    if lb is None and ub is None: return None
    if lb is None: return ub
    if ub is None: return lb
    try:
        return (float(lb) + float(ub)) / 2.0
    except Exception:
        return None

def flatten_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tabellenfreundliche Sicht auf wichtige Felder."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        imp = r.get("impressions") or {}
        sp  = r.get("spend") or {}
        languages = ", ".join(r.get("languages") or [])
        platforms = ", ".join(r.get("publisher_platforms") or [])
        d: Dict[str, Any] = {
            "id": r.get("id"),
            "page_name": r.get("page_name"),
            "page_id": r.get("page_id"),
            "ad_creation_time": r.get("ad_creation_time"),
            "ad_delivery_start_time": r.get("ad_delivery_start_time"),
            "ad_delivery_stop_time": r.get("ad_delivery_stop_time"),
            "languages": languages or None,
            "platforms": platforms or None,
            "currency": r.get("currency"),
            "impressions_lower": _as_int(imp.get("lower_bound")),
            "impressions_upper": _as_int(imp.get("upper_bound")),
            "impressions_mid": _mid(_as_int(imp.get("lower_bound")), _as_int(imp.get("upper_bound"))),
            "spend_lower": _as_int(sp.get("lower_bound")),
            "spend_upper": _as_int(sp.get("upper_bound")),
            "spend_mid": _mid(_as_int(sp.get("lower_bound")), _as_int(sp.get("upper_bound"))),
            "ad_snapshot_url": r.get("ad_snapshot_url"),
            "demographic_distribution": json.dumps(r.get("demographic_distribution"), ensure_ascii=False) if r.get("demographic_distribution") else None,
            "delivery_by_region": json.dumps(r.get("delivery_by_region"), ensure_ascii=False) if r.get("delivery_by_region") else None,
            "total_reach_by_location": json.dumps(r.get("total_reach_by_location"), ensure_ascii=False) if r.get("total_reach_by_location") else None,
        }
        out.append(d)
    return out

# ------------------------------------------------------------
# Hauptfunktion für UI/CLI
# ------------------------------------------------------------
def run_search(
    *,
    topic: str,
    country: str,
    date_from: dt.date,
    date_to: dt.date,
    per_lang: int = 12,
    write_db: bool = True,
    progress: ProgressFn = None,
    forced_keywords: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    if not topic or not country or not date_from or not date_to:
        raise ValueError("topic, country, date_from, date_to sind Pflicht.")

    access_token = os.getenv("META_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("ENV META_ACCESS_TOKEN fehlt.")

    campaign_name = topic.strip()
    campaign_slug = _slugify(campaign_name)
    output_dir = campaign_name

    # Keywords
    if progress: progress("Erzeuge Keywords …", 0.05)
    if forced_keywords:
        kw_by_lang = {k: (forced_keywords.get(k) or [])[:per_lang] for k in ["de", "fr", "it"]}
        for lang in ["de", "fr", "it"]:
            while len(kw_by_lang[lang]) < per_lang:
                kw_by_lang[lang].append(f"{topic.strip()} kw{len(kw_by_lang[lang])+1}")
        # DE: topic sicherstellen
        if topic.strip().lower() not in [x.lower() for x in kw_by_lang["de"]]:
            kw_by_lang["de"] = [topic.strip()] + kw_by_lang["de"]
            kw_by_lang["de"] = kw_by_lang["de"][:per_lang]
    else:
        kw_by_lang = generate_keywords(topic, per_lang=per_lang)

    keywords: List[str] = []
    for lang in ["de", "fr", "it"]:
        keywords.extend(kw_by_lang.get(lang, []))

    # API-Params
    base_params: Dict[str, str] = {
        "ad_type": "POLITICAL_AND_ISSUE_ADS",
        "ad_active_status": "ALL",
        "ad_reached_countries": country,
        "search_type": "KEYWORD_EXACT_PHRASE",
        "limit": "500",
        "fields": FIELDS,
        "access_token": access_token,
        "ad_delivery_date_min": date_from.isoformat(),
        "ad_delivery_date_max": date_to.isoformat(),
    }

    # Suchen
    combined: List[Dict] = []
    total_q = max(1, len(keywords))
    for i, q in enumerate(keywords, 1):
        if progress:
            progress(f"Suche {i}/{total_q}: {q}", 0.06 + 0.74 * (i / total_q))
        rows = _fetch_one_query(base_params, q)
        combined.extend(rows)

    deduped = _dedupe_by_id(combined)
    if progress: progress(f"Dedupliziert: {len(deduped)} Ads", 0.83)

    out_path = _filename_for_today(output_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    # DB (optional)
    db_info: Dict[str, Any] = {"written": False, "inserted_ads": 0}
    snapshot_date = _today_str()
    file_url = Path(os.path.abspath(out_path)).resolve().as_uri()

    if write_db and _HAVE_DB and os.getenv("DATABASE_URL"):
        try:
            if progress: progress("Schreibe in DB …", 0.92)

            # --- Timings robust initialisieren (gegen Pylance-Warnungen) ---
            t0 = time.perf_counter()
            t1 = t0
            t2 = t0

            with connect() as conn:
                # Session-Guards (falls du _configure_pg_session eingebaut hast)
                try:
                    _configure_pg_session(conn, app_name="meta_api_agent")
                except Exception:
                    pass

                # 1) Kampagne/Snapshot
                campaign_id = get_or_create_campaign(conn, campaign_name, campaign_slug)
                save_api_snapshot(conn, campaign_id, snapshot_date, deduped, file_url)
                t1 = time.perf_counter()

                # 2) Ads upserten – mit Batches und Fortschritt
                total = len(deduped)
                inserted = 0
                BATCH = 500

                for i, r in enumerate(deduped, 1):
                    try:
                        upsert_ad_with_raw(conn, campaign_id, r)
                        inserted += 1
                    except Exception as inner_e:
                        # Wichtig: Rollback, sonst "idle in transaction"
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        print(f"   ⚠️ konnte Ad nicht speichern (id={r.get('id')}): {inner_e}")

                    if i % BATCH == 0:
                        try:
                            conn.commit()
                        except Exception:
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                        if progress:
                            p = 0.92 + 0.07 * (i / max(1, total))
                            progress(f"Schreibe in DB … {i}/{total}", min(0.99, p))

                try:
                    conn.commit()
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            t2 = time.perf_counter()

            if progress:
                snap_dur = t1 - t0
                ads_dur  = t2 - t1
                progress(
                    f"DB fertig: Snapshot {snap_dur:.1f}s, Ads {ads_dur:.1f}s (insg. {inserted}/{len(deduped)})",
                    1.0,
                )

            db_info = {"written": True, "inserted_ads": inserted, "snapshot_date": snapshot_date}
        except Exception as e:
            db_info = {"written": False, "error": str(e)}

    if progress: progress("Fertig.", 1.0)

    return {
        "campaign_name": campaign_name,
        "campaign_slug": campaign_slug,
        "used_keywords": {"de": kw_by_lang.get("de", []),
                          "fr": kw_by_lang.get("fr", []),
                          "it": kw_by_lang.get("it", [])},
        "results": deduped,
        "saved_file": out_path,
        "db": db_info,
        "country": country,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
    }

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--country", default="CH")
    ap.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    ap.add_argument("--per-lang", type=int, default=12)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    df = dt.date.fromisoformat(args.date_from)
    dt_ = dt.date.fromisoformat(args.date_to)

    def _p(msg, p): print(f"[{int(p*100):3d}%] {msg}")

    out = run_search(
        topic=args.topic,
        country=args.country,
        date_from=df, date_to=dt_,
        per_lang=args.per_lang,
        write_db=not args.no_db,
        progress=_p,
    )
    print(f"\n✓ {len(out['results'])} Ads, Datei: {out['saved_file']}")
    if out["db"].get("written"):
        print(f"DB: {out['db']['inserted_ads']} Ads gespeichert (Snapshot {out['db']['snapshot_date']})")
