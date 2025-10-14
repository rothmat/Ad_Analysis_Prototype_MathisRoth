"""
Meta Ad Library Tracker – Kampagne "Eigenmietwert" (CH)
- Sucht täglich alle politischen/issue Ads zum Themenkomplex "Eigenmietwert".
- Speichert alle Treffer eines Tages in ./Eigenmietwert/YYYY-MM-DD.json
- Paginierung, Deduplikation, defensives Error-Handling

+ NEU: schreibt zusätzlich den Tages-Snapshot in Postgres (api_snapshots)
+ NEU: schreibt jede einzelne Ad (mit allen Feldern) in ads.raw (JSONB) inkl. campaign_id
Getestet mit Graph API v23.0
"""

import os
import json
import time
import datetime as dt
from typing import Dict, List, Optional
import requests
from pathlib import Path
import sys

# --- db_client aus ../ad-db/ingest laden (Sibling-Folder) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ad-db" / "ingest"))
from db_client import (  # type: ignore
    connect,
    get_or_create_campaign,
    save_api_snapshot,
    upsert_ad_with_raw,   # NEU: speichert vollständige Ad in ads.raw
)

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
ACCESS_TOKEN = "EAARLKoboLQwBPCHkwoxuvF4UT5C23dedwC4nogXzxqkmZBhZAaQ2jtnmYsriOqqMjWE7VEnxwZBxfP4RaDhXbXjEOowuG2osMgbmiKHNWpE50w1TJ4NuFBzLNCMdvG1VDzgGh3jtroNWqIAC6JAJeGrBQdfSQznaED2Y3S0RP3byuHn7CnMi4CIZAmAjEKnmAucZCQRYDDdod"
BASE_URL = "https://graph.facebook.com/v23.0/ads_archive"

# Kampagnen-Ordner/Name
CAMPAIGN_NAME = "Eigenmietwert"
OUTPUT_DIR = CAMPAIGN_NAME
CAMPAIGN_SLUG = CAMPAIGN_NAME.lower().replace(" ", "_")  # für DB

# Land/Region
COUNTRY = "CH"

# Suchphrasen
KEYWORDS = [
    # --- Deutsch ---
    "Eigenmietwert",
    "Abschaffung Eigenmietwert",
    "Besteuerung Eigenmietwert",
    "Eigenmietwert Steuer",
    "Wohneigentum Eigenmietwert",
    "Systemwechsel Eigenmietwert",
    "Eigenmietwert abschaffen",
    "Steuer auf Eigenmietwert",
    "Wohnsteuer Schweiz",
    "Hausbesitzer Steuer",
    "Wohneigentum fördern",
    "Systemwechsel Wohneigentum",
    "Immobilienbesteuerung Schweiz",
    # --- Französisch ---
    "valeur locative",
    "suppression valeur locative",
    "imposition valeur locative",
    "valeur locative logement",
    "système valeur locative",
    "suppression de la valeur locative",
    "imposition de la valeur locative",
    "logement valeur locative",
    "propriétaires immobiliers impôt",
    "réforme valeur locative",
    "système fiscal logement",
    "impôt sur la valeur locative",
    # --- Italienisch ---
    "valore locativo",
    "abolizione valore locativo",
    "tassazione valore locativo",
    "valore locativo abitazione",
    "sistema valore locativo",
    "abolizione del valore locativo",
    "tassazione del valore locativo",
    "casa valore locativo",
    "proprietari di casa imposta",
    "riforma valore locativo",
    "sistema fiscale abitazione",
    "imposta sul valore locativo",
]

FIELDS = ",".join([
    "id","ad_creation_time","ad_delivery_start_time","ad_delivery_stop_time",
    "ad_creative_bodies","ad_creative_link_captions","ad_creative_link_descriptions",
    "ad_creative_link_titles","ad_snapshot_url","currency","demographic_distribution",
    "delivery_by_region","impressions","page_id","page_name","publisher_platforms",
    "spend","languages","bylines","estimated_audience_size","age_country_gender_reach_breakdown",
    "beneficiary_payers","eu_total_reach","target_ages","target_gender","target_locations",
    "total_reach_by_location",
])

BASE_PARAMS: Dict[str, str] = {
    "ad_type": "POLITICAL_AND_ISSUE_ADS",
    "ad_active_status": "ALL",
    "ad_reached_countries": COUNTRY,
    "search_type": "KEYWORD_EXACT_PHRASE",
    "limit": "500",
    "fields": FIELDS,
    "access_token": ACCESS_TOKEN,
}

REQUEST_TIMEOUT = 40
RETRY_BACKOFFS = [2, 4, 8, 16]  # Sekunden

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _today_str(tz: Optional[dt.tzinfo] = None) -> str:
    return dt.datetime.now(tz).strftime("%Y-%m-%d")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _filename_for_today() -> str:
    _ensure_dir(OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, f"{_today_str()}.json")

def _get_with_retries(url: str, params: Dict[str, str]) -> requests.Response:
    last_exc = None
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

def _fetch_one_query(query: str) -> List[Dict]:
    params = dict(BASE_PARAMS)
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
        rid = str(r.get("id"))
        if not rid:
            continue
        seen[rid] = r
    return list(seen.values())

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_once() -> str:
    print(f"→ Starte Abruf für Kampagne: {CAMPAIGN_NAME} (Land={COUNTRY})")
    combined: List[Dict] = []

    for q in KEYWORDS:
        print(f"   • Suche: {q}")
        rows = _fetch_one_query(q)
        print(f"     → {len(rows)} Treffer")
        combined.extend(rows)

    deduped = _dedupe_by_id(combined)
    print(f"✓ Dedupliziert: {len(deduped)} eindeutige Ads")

    # 1) Datei schreiben (wie bisher)
    out_path = _filename_for_today()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"✅ Gespeichert: {out_path}")

    # 2) in DB speichern (api_snapshots + jede Ad in ads.raw)
    snapshot_date = _today_str()
    file_url = Path(os.path.abspath(out_path)).resolve().as_uri()

    try:
        with connect() as conn:
            campaign_id = get_or_create_campaign(conn, CAMPAIGN_NAME, CAMPAIGN_SLUG)

            # Kampagnen-Snapshot speichern
            save_api_snapshot(conn, campaign_id, snapshot_date, deduped, file_url)

            # Alle Ads als einzelne Zeilen in 'ads' (mit vollständigem JSON unter ads.raw)
            inserted = 0
            for r in deduped:
                try:
                    upsert_ad_with_raw(conn, campaign_id, r)
                    inserted += 1
                except Exception as inner_e:
                    print(f"   ⚠️  konnte Ad nicht speichern (id={r.get('id')}): {inner_e}")

            conn.commit()
        print(f"🗄️  DB: api_snapshots + {inserted} ads (raw) für {CAMPAIGN_NAME} @ {snapshot_date}")
    except Exception as e:
        # Lauf nicht hart failen, falls DB gerade nicht erreichbar ist.
        print(f"⚠️  Konnte nicht in DB schreiben: {e}")

    return out_path

if __name__ == "__main__":
    run_once()
