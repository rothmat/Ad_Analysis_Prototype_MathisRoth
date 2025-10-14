# -*- coding: utf-8 -*-
"""
results_merge.py
----------------
Mergt Facebook-API Ads NUR mit vorhandenen LLM (OpenAI) Analysen und speichert pro Ad ein
zusammengeführtes JSON in einer separaten Tabelle (ad_llm_fused).
Zusätzlich wird die gesamte Liste der "fused"-Objekte lokal unter
<CampaignRoot>/Api-LLM-Fused/<YYYY-MM-DD>.json gespeichert.

ENV:
  DATABASE_URL = postgresql://app:app@localhost:5432/appdb

Aufruf-Beispiel (PowerShell):
  py .\agents\Data-Preparation\results_merge.py ^
    --campaign-slug eigenmietwert ^
    --api-json .\Eigenmietwert\2025-09-03.json ^
    --llm-json .\Eigenmietwert\LLM-Analysis\OpenAI\2025-09-03.json ^
    --snapshot-date 2025-09-03
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, date
from decimal import Decimal

# db_client import analog zu deinen Agents
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad-db" / "ingest"))
from db_client import connect  # type: ignore


# ---------- JSON Helper (Dates/Decimals serialisieren) ----------

def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)


# ---------- Helpers: Datei-Lader ----------

def load_json_any(path: Path) -> List[Dict[str, Any]]:
    """
    Lädt entweder JSON-Array, JSON-Objekt (mit .get('data', [])) oder NDJSON.
    Gibt Liste von Dicts zurück.
    """
    text = path.read_text(encoding="utf-8")
    out: List[Dict[str, Any]] = []
    # probiere JSON
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
    # NDJSON
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


# ---------- DB-Queries/Upserts ----------

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


def get_campaign(conn, slug: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT id, name, slug FROM campaigns WHERE slug=%s"
    with conn.cursor() as cur:
        cur.execute(sql, (slug,))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "slug": row[2]}


def get_ad_by_external_id(conn, campaign_id: int, ad_external_id: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT id, ad_external_id FROM ads WHERE campaign_id=%s AND ad_external_id=%s"
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_id, ad_external_id))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "ad_external_id": row[1]}


def media_ad_id_for_media_id(conn, media_id: int) -> Optional[int]:
    sql = "SELECT ad_id FROM media WHERE id=%s"
    with conn.cursor() as cur:
        cur.execute(sql, (media_id,))
        r = cur.fetchone()
        return int(r[0]) if r else None


def get_media_for_ad(conn, ad_id: int, date_folder: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Liefert Medienlisten (screenshots/images/videos) für ad_id; optional gefiltert nach date_folder (YYYY-MM-DD).
    """
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
            # df kann ein datetime.date sein -> in ISO-String wandeln
            if isinstance(df, (date, datetime)):
                df = df.isoformat()
            item = {"media_id": mid, "file_url": file_url, "filename": filename, "date_folder": df}
            if kind == "screenshot":
                out["screenshots"].append(item)
            elif kind == "image":
                out["images"].append(item)
            elif kind == "video":
                out["videos"].append(item)
    return out


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
    return rid


# ---------- LLM-Mapping (LLM JSON -> ad_id) ----------

def parse_media_id_from_screenshot_id(sid: str) -> Optional[int]:
    # erwartet "media:<id>"
    if not isinstance(sid, str):
        return None
    sid = sid.strip()
    if sid.startswith("media:"):
        try:
            return int(sid.split(":", 1)[1])
        except Exception:
            return None
    return None


def index_llm_by_ad(conn, llm_items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Baut Mapping ad_id -> bester LLM-Eintrag.
    Auswahlregel:
      - bevorzuge geeignet=true
      - wenn mehrere geeignet=true: nimm den ersten im File
      - sonst (nur ungeeignet vorhanden): nimm den ersten ungeeigneten
    """
    best: Dict[int, Dict[str, Any]] = {}

    for item in llm_items:
        sid = item.get("screenshot_id")
        mid = parse_media_id_from_screenshot_id(sid) if sid else None
        if mid is None:
            continue
        ad_id = media_ad_id_for_media_id(conn, mid)
        if ad_id is None:
            continue

        suitable = bool(item.get("geeignet") is True)

        # falls schon geeigneter existiert, weiter
        if ad_id in best and best[ad_id].get("geeignet") is True:
            continue

        # falls noch kein Eintrag oder der vorhandene ungeeignet ist, ersetze mit geeignetem
        if (ad_id not in best) or suitable:
            best[ad_id] = item

    return best


# ---------- Media-Filter: nur LLM-Screenshot behalten ----------

def filter_media_to_llm_screenshot(media_refs: Dict[str, List[Dict[str, Any]]],
                                   llm_item: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reduziert media_refs auf genau den Screenshot, der im LLM-Item verwendet wurde.
    Löscht images/videos, denn diese wurden nicht analysiert.
    """
    if not llm_item:
        return media_refs  # kein LLM -> nichts filtern

    sid = llm_item.get("screenshot_id")
    mid = parse_media_id_from_screenshot_id(sid) if sid else None
    if mid is None:
        # Kein gültiger media:<id> -> leeres Media-Set zurückgeben
        print(f"⚠️  Ungültige oder fehlende screenshot_id im LLM-Item: {sid}")
        return {"screenshots": [], "images": [], "videos": []}

    only_shot = [m for m in media_refs.get("screenshots", []) if int(m.get("media_id", -1)) == int(mid)]
    if not only_shot:
        # Hinweis: Der Aufrufer kann ggf. einen zweiten Versuch ohne Datumsfilter machen.
        print(f"⚠️  LLM-Screenshot media_id={mid} nicht im aktuellen Medien-Set gefunden.")
    return {"screenshots": only_shot, "images": [], "videos": []}


# ---------- Merging ----------

def make_fused_json(now_iso: str,
                    campaign: Dict[str, Any],
                    ad_pk: int,
                    ad_external_id: str,
                    snapshot_date: str,
                    api_ad: Dict[str, Any],
                    llm_item: Optional[Dict[str, Any]],
                    media_refs: Dict[str, List[Dict[str, Any]]]
                    ) -> Dict[str, Any]:
    """
    Baut das Ziel-JSON.
    """
    api_block = {
        "source": "facebook_ad_library",
        "raw": api_ad
    }

    llm_block = None
    if llm_item is not None:
        llm_block = {
            "provider": "openai",
            "analysis_file_payload": llm_item
        }

    fused = {
        "version": "1.0",
        "merge_time": now_iso,
        "ad_keys": {
            "ad_external_id": ad_external_id,
            "ad_pk": ad_pk,
            "campaign": {"id": campaign["id"], "slug": campaign["slug"], "name": campaign["name"]},
            "snapshot_date": snapshot_date
        },
        "api": api_block,
        "media": {
            "screenshots": media_refs.get("screenshots", []),
            "images": media_refs.get("images", []),
            "videos": media_refs.get("videos", []),
        },
        "llm_analysis": llm_block,
        "derived": {
            "has_llm": llm_block is not None,
            "has_video": len(media_refs.get("videos", [])) > 0
        }
    }
    return fused


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-slug", required=True, help="z.B. eigenmietwert")
    ap.add_argument("--api-json", required=True, help="Pfad zur API-Ergebnisdatei (JSON/NDJSON)")
    ap.add_argument("--llm-json", required=True, help="Pfad zur OpenAI LLM-JSON-Datei (…/LLM-Analysis/OpenAI/DATE.json)")
    ap.add_argument("--snapshot-date", help="YYYY-MM-DD (für DB-Schlüssel/Medien-Filter und lokalen Dateinamen).")
    args = ap.parse_args()

    api_path = Path(args.api_json)
    llm_path = Path(args.llm_json)

    api_ads = load_json_any(api_path)
    llm_items = load_json_any(llm_path)

    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Lokal-Datei vorbereiten (Api-LLM-Fused/<DATE>.json)
    file_date = args.snapshot_date or date.today().isoformat()
    campaign_root = Path.cwd() / args.campaign_slug.capitalize()
    fused_dir = campaign_root / "Api-LLM-Fused"
    fused_dir.mkdir(parents=True, exist_ok=True)
    fused_out_path = fused_dir / f"{file_date}.json"

    file_outputs: List[Dict[str, Any]] = []

    with connect() as conn:
        ensure_fused_table(conn)

        campaign = get_campaign(conn, args.campaign_slug)
        if not campaign:
            print(f"❌ Kampagne '{args.campaign_slug}' nicht gefunden.")
            return

        # LLM-Index: ad_id -> LLM-Item
        llm_by_ad = index_llm_by_ad(conn, llm_items)

        merged_ok = 0
        missing_ads = 0
        skipped_no_llm = 0

        for api_ad in api_ads:
            ad_external_id = str(api_ad.get("id") or "").strip()
            if not ad_external_id:
                continue

            ad_row = get_ad_by_external_id(conn, campaign["id"], ad_external_id)
            if not ad_row:
                missing_ads += 1
                continue

            ad_pk = int(ad_row["id"])

            # Nur mergen, wenn LLM-Analyse vorhanden
            llm_item = llm_by_ad.get(ad_pk)
            if llm_item is None:
                skipped_no_llm += 1
                continue

            # Snapshot-Datum: Vorrang CLI, dann API-Feld, sonst heute
            snapshot_date = (args.snapshot_date
                             or api_ad.get("ad_creation_time")
                             or api_ad.get("ad_delivery_start_time")
                             or date.today().isoformat())
            snapshot_date = str(snapshot_date)[:10]

            # 1. Medien nach Datum holen
            media_refs = get_media_for_ad(conn, ad_pk, date_folder=snapshot_date)

            # 2. Nur LLM-Screenshot behalten
            filtered = filter_media_to_llm_screenshot(media_refs, llm_item)
            if not filtered["screenshots"]:
                # 3. Fallback: ohne Datumsfilter alle Medien holen und erneut filtern
                all_media = get_media_for_ad(conn, ad_pk, date_folder=None)
                filtered = filter_media_to_llm_screenshot(all_media, llm_item)

            media_refs = filtered

            fused = make_fused_json(
                now_iso=now_iso,
                campaign=campaign,
                ad_pk=ad_pk,
                ad_external_id=ad_external_id,
                snapshot_date=snapshot_date,
                api_ad=api_ad,
                llm_item=llm_item,
                media_refs=media_refs
            )

            rid = upsert_fused(conn, ad_pk, snapshot_date, fused)
            merged_ok += 1
            file_outputs.append(fused)
            print(f"  ✓ fused gespeichert (DB row_id={rid}, ad_pk={ad_pk}, ad_id={ad_external_id})")

    # Lokale Sammeldatei schreiben
    try:
        with open(fused_out_path, "w", encoding="utf-8") as f:
            json.dump(file_outputs, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"\n🗂  Datei geschrieben: {fused_out_path}")
    except Exception as e:
        print(f"⚠️  Konnte lokale Datei nicht schreiben: {e}")

    print("\n— Merge fertig —")
    print(f"  API Ads gelesen:            {len(api_ads)}")
    print(f"  LLM Items gelesen:          {len(llm_items)}")
    print(f"  Erfolgreich gemerged:       {len(file_outputs)}")
    print(f"  Übersprungen (kein LLM):    {skipped_no_llm}")
    print(f"  Nicht in DB gefundene Ads:  {missing_ads}")


if __name__ == "__main__":
    main()
