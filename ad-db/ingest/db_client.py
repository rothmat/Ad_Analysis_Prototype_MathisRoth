from __future__ import annotations
import os, json, hashlib, base64, mimetypes
from pathlib import Path
from typing import Optional, Dict, Any
import psycopg

DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/appdb")

def connect():
    """Postgres-Verbindung öffnen. Mit `with connect() as conn:` nutzen."""
    return psycopg.connect(DB_URL)

# -------------------- Helpers --------------------

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_ads_raw_column(conn) -> None:
    """Stellt sicher, dass ads.raw (JSONB) existiert. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE ads
            ADD COLUMN IF NOT EXISTS raw JSONB;
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ads_raw_gin
            ON ads USING GIN (raw jsonb_path_ops);
        """)

def _ensure_media_base64_table(conn) -> None:
    """Legt die Tabelle für Base64-Daten an (falls nicht vorhanden)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS media_base64 (
              media_id BIGINT PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
              mime_type TEXT,
              b64 TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

# -------------------- Stammdaten --------------------

def get_or_create_campaign(conn, name: str, slug: Optional[str] = None) -> int:
    slug = slug or name.lower().replace(" ", "_")
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO campaigns(name, slug)
          VALUES (%s,%s)
          ON CONFLICT (slug) DO UPDATE SET name=EXCLUDED.name
          RETURNING id
        """, (name, slug))
        return cur.fetchone()[0]

def upsert_ad(conn, campaign_id: int, ad_external_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO ads(campaign_id, ad_external_id, first_seen, last_seen)
          VALUES (%s,%s, now(), now())
          ON CONFLICT (campaign_id, ad_external_id) DO UPDATE
          SET last_seen = GREATEST(ads.last_seen, EXCLUDED.last_seen)
          RETURNING id
        """, (campaign_id, ad_external_id))
        return cur.fetchone()[0]

def upsert_ad_with_raw(conn, campaign_id: int, ad: Dict[str, Any]) -> int:
    """
    Legt/aktualisiert eine Ad (unique per campaign_id + ad_external_id) und speichert
    den kompletten Original-Record in ads.raw (JSONB).
    """
    ad_external_id = str(ad.get("id") or "").strip()
    if not ad_external_id:
        raise ValueError("ad ohne 'id' im Payload")

    _ensure_ads_raw_column(conn)

    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO ads (campaign_id, ad_external_id, first_seen, last_seen, raw)
          VALUES (%s, %s, now(), now(), %s)
          ON CONFLICT (campaign_id, ad_external_id) DO UPDATE
          SET last_seen = GREATEST(ads.last_seen, EXCLUDED.last_seen),
              raw       = EXCLUDED.raw
          RETURNING id
        """, (campaign_id, ad_external_id, json.dumps(ad)))
        return cur.fetchone()[0]

def get_ad_id(conn, campaign_slug: str, ad_external_id: str) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute("""
          SELECT ad.id
          FROM ads ad
          JOIN campaigns c ON c.id = ad.campaign_id
          WHERE c.slug=%s AND ad.ad_external_id=%s
          LIMIT 1
        """, (campaign_slug, ad_external_id))
        row = cur.fetchone()
        return row[0] if row else None

# -------------------- API-/Ad-JSON --------------------

def save_api_snapshot(conn, campaign_id: int, snapshot_date: str, payload: Dict[str, Any] | list, file_url: Optional[str]=None) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO api_snapshots(campaign_id, snapshot_date, payload, file_url)
          VALUES (%s,%s,%s,%s)
          ON CONFLICT (campaign_id, snapshot_date) DO UPDATE
          SET payload = EXCLUDED.payload,
              file_url = COALESCE(EXCLUDED.file_url, api_snapshots.file_url)
          RETURNING id
        """, (campaign_id, snapshot_date, json.dumps(payload), file_url))
        return cur.fetchone()[0]

def save_ad_snapshot(conn, ad_id: int, snapshot_date: str, payload: Dict[str, Any], file_url: Optional[str]=None) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO ad_snapshots(ad_id, snapshot_date, payload, file_url)
          VALUES (%s,%s,%s,%s)
          ON CONFLICT (ad_id, snapshot_date) DO UPDATE
          SET payload = EXCLUDED.payload,
              file_url = COALESCE(EXCLUDED.file_url, ad_snapshots.file_url)
          RETURNING id
        """, (ad_id, snapshot_date, json.dumps(payload), file_url))
        return cur.fetchone()[0]

# -------------------- Medien --------------------

def add_media_local(conn, ad_id: int, path: str, kind: str, date_folder: Optional[str]=None,
                    width: Optional[int]=None, height: Optional[int]=None, duration_seconds: Optional[float]=None) -> Optional[int]:
    """
    Legt einen Media-Datensatz an (mit lokalem file:// Pfad). Idempotent (sha256).
    kind: 'image' | 'video' | 'screenshot'
    Gibt IMMER die media.id zurück (auch wenn bereits existent).
    """
    p = Path(path)
    if not p.exists():
        return None
    sha = _sha256_file(p)
    file_url = p.resolve().as_uri()
    with conn.cursor() as cur:
        # Einmalig empfohlen (außerhalb): ALTER TABLE media ADD CONSTRAINT uq_media_ad_sha UNIQUE (ad_id, sha256);
        cur.execute("""
          INSERT INTO media(ad_id, kind, format, file_url, filename, date_folder,
                            bytes, width, height, duration_seconds, sha256)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
          ON CONFLICT DO NOTHING
          RETURNING id
        """, (
            ad_id, kind, p.suffix.lstrip("."),
            file_url, p.name, date_folder,
            p.stat().st_size, width, height, duration_seconds, sha
        ))
        row = cur.fetchone()
        if row and row[0]:
            return row[0]

        # Falls es schon existiert (ON CONFLICT DO NOTHING), alte id nachschlagen:
        cur.execute("""
          SELECT id FROM media
          WHERE ad_id=%s AND sha256=%s
          ORDER BY id DESC
          LIMIT 1
        """, (ad_id, sha))
        row2 = cur.fetchone()
        return row2[0] if row2 else None

def upsert_media_base64_from_path(conn, media_id: int, file_path: str, mime_type: Optional[str] = None) -> int:
    """
    Liest die Datei, erstellt Base64 und speichert/aktualisiert in media_base64.
    """
    _ensure_media_base64_table(conn)
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)

    if not mime_type:
        mime_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO media_base64(media_id, mime_type, b64)
          VALUES (%s,%s,%s)
          ON CONFLICT (media_id) DO UPDATE
          SET b64 = EXCLUDED.b64,
              mime_type = COALESCE(EXCLUDED.mime_type, media_base64.mime_type)
          RETURNING media_id
        """, (media_id, mime_type, b64))
        return cur.fetchone()[0]

# -------------------- Analysen (für später) --------------------

def upsert_analysis(conn, name: str, provider: str, version: str, parameters: Optional[Dict[str, Any]]=None) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO analyses(name, provider, version, parameters)
          VALUES (%s,%s,%s,%s)
          ON CONFLICT (name, provider, version) DO UPDATE
          SET parameters = EXCLUDED.parameters
          RETURNING id
        """, (name, provider, version, json.dumps(parameters or {})))
        return cur.fetchone()[0]

def save_result_for_ad(conn, analysis_id: int, ad_id: int, result: Dict[str, Any], score: Optional[float]=None) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO analysis_results(analysis_id, target_type, ad_id, result, score)
          VALUES (%s,'ad',%s,%s,%s)
          RETURNING id
        """, (analysis_id, ad_id, json.dumps(result), score))
        return cur.fetchone()[0]

def save_result_for_campaign(conn, analysis_id: int, campaign_id: int, result: Dict[str, Any], score: Optional[float]=None) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO analysis_results(analysis_id, target_type, campaign_id, result, score)
          VALUES (%s,'campaign',%s,%s,%s)
          RETURNING id
        """, (analysis_id, campaign_id, json.dumps(result), score))
        return cur.fetchone()[0]

def link_ensemble(conn, ensemble_result_id: int, openai_result_id: Optional[int], gemini_result_id: Optional[int]) -> None:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO ensemble_links(ensemble_result_id, source_openai_result_id, source_gemini_result_id)
          VALUES (%s,%s,%s)
          ON CONFLICT (ensemble_result_id) DO UPDATE
          SET source_openai_result_id = EXCLUDED.source_openai_result_id,
              source_gemini_result_id = EXCLUDED.source_gemini_result_id
        """, (ensemble_result_id, openai_result_id, gemini_result_id))
