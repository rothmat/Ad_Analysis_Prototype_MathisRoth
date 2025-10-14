# agents/ad_tagger.py
import json
from typing import Any, Dict, List, Optional, Callable, Tuple

import pandas as pd

# DB-Connect ausschließlich über unseren robusten Loader
from agents._db import connect

# ---------------- Prompt / Parsing ----------------
SYSTEM_PROMPT = """Du bist Analyst:in für politische Online-Werbung.

ZIEL
- Klassifiziere EINE Anzeige (Ad-Level) in bis zu 3 thematische Felder.
- Liefere eine numerische Confidence und kurze, stichpunktartige Begründungen.
- Gib AUSSCHLIESSLICH kompaktes JSON zurück (kein Markdown, keine Erklärungen).

ONTOLOGIE (Themen; benutze NUR diese Strings, Reihenfolge = Relevanz, max. 3)
["Klima & Energie","Mobilität & ÖV","Soziales & Verteilung","Migration & Integration",
 "Sicherheit & Ordnung","Wirtschaft & Innovation","Außen & Entwicklung","Sonstiges"]

HINWEISE
- Nutze Text (Headline/Body), CTA-Ton, Claims und evtl. Plattformhinweise.
- Wenn unklar, nutze genau "Sonstiges".
- Confidence (0.0–1.0):
  0.2–0.4 sehr wenig Signal; 0.5–0.7 klare(n) Hauptindikator(en); 0.75–0.9 mehrere starke Hinweise; 0.95 nur bei Eindeutigkeit.
- Begründungen als kurze Bullets (max. 4), z.B.: „fordert Abschaffung X“, „spricht Steuerzahler an“, „Framing: Sicherheit“.

AUSGABE (nur JSON, exakt dieses Schema; Arrays ohne Duplikate):
{
  "id": "<AD_ID>",
  "topics": ["<Thema1>","<Thema2>"],
  "confidence": 0.78,
  "rationale_bullets": ["<kurz>","<kurz>","<kurz>"]
}
"""

def _json_first(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    if "{" in s:
        s = s[s.index("{"):]
    try:
        return json.loads(s)
    except Exception:
        return {}

# ---------------- DB: Tabellen & Queries ----------------
def ensure_topics_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ad_topics_results (
            ad_id              INTEGER PRIMARY KEY,
            campaign_id        INTEGER,
            campaign_slug      TEXT,
            page_name          TEXT,
            bylines            TEXT[],
            media_id           TEXT,
            topics             JSONB,
            rationale_bullets  JSONB,
            confidence         DOUBLE PRECISION,
            model              TEXT,
            analyzed_at        TIMESTAMPTZ DEFAULT now()
        )""")
    conn.commit()

def list_campaigns(conn) -> List[Tuple[int,str,str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, slug FROM campaigns ORDER BY name")
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

def list_fused_ads_for_campaign(conn, campaign_slug: str) -> pd.DataFrame:
    sql = """
      SELECT DISTINCT a.id AS ad_pk,
             a.ad_external_id,
             COALESCE((f.fused->'api'->'raw'->>'page_name'),'')        AS page_name,
             COALESCE((f.fused->'api'->'raw'->>'bylines'),'')          AS bylines_raw,
             COALESCE((f.fused->'media'->'screenshots'->0->>'media_id'),'') AS media_id
      FROM ad_llm_fused f
      JOIN ads a ON a.id = f.ad_id
      JOIN campaigns c ON c.id = a.campaign_id
      WHERE c.slug=%s
      ORDER BY a.id DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug,))
        rows = cur.fetchall()

    def _bylines(v: str) -> List[str]:
        if not v:
            return []
        if v.strip().startswith("["):
            try:
                return [x for x in json.loads(v) if str(x).strip()]
            except:
                return [v]
        return [v]

    out = []
    for ad_pk, ad_ext, page_name, by_raw, media_id in rows:
        mid = str(media_id or ad_ext or "").strip()
        pg  = (page_name or "").strip() or "Unbekannte Gruppe"
        label = f"{pg} ({mid})" if mid else pg  # <<< gewünschtes Format

        out.append({
            "ad_pk": int(ad_pk),
            "ad_external_id": ad_ext,
            "page_name": pg,
            "bylines": _bylines(by_raw),
            "media_id": mid,
            "label": label,
        })
    return pd.DataFrame(out)

def _already_done(conn, ad_pk: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM ad_topics_results WHERE ad_id=%s LIMIT 1", (ad_pk,))
        return cur.fetchone() is not None

def _upsert_row(conn, row: Dict[str, Any]) -> None:
    sql = """
    INSERT INTO ad_topics_results
      (ad_id,campaign_id,campaign_slug,page_name,bylines,media_id,topics,rationale_bullets,confidence,model,analyzed_at)
    VALUES
      (%(ad_id)s,%(campaign_id)s,%(campaign_slug)s,%(page_name)s,%(bylines)s,%(media_id)s,
       %(topics)s,%(rationale_bullets)s,%(confidence)s,%(model)s,now())
    ON CONFLICT (ad_id) DO UPDATE SET
      topics=EXCLUDED.topics,
      rationale_bullets=EXCLUDED.rationale_bullets,
      confidence=EXCLUDED.confidence,
      page_name=EXCLUDED.page_name,
      bylines=EXCLUDED.bylines,
      media_id=EXCLUDED.media_id,
      model=EXCLUDED.model,
      analyzed_at=now()
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)

def _fetch_fused_rows(conn, campaign_slug: str, ad_pks: List[int]) -> List[Dict[str, Any]]:
    if not ad_pks:
        return []
    sql = """
      SELECT a.id AS ad_pk, a.campaign_id, c.slug, f.fused,
             COALESCE((f.fused->'api'->'raw'->>'page_name'),'')        AS page_name,
             COALESCE((f.fused->'api'->'raw'->>'bylines'),'')          AS bylines_raw,
             COALESCE((f.fused->'media'->'screenshots'->0->>'media_id'),'') AS media_id
      FROM ad_llm_fused f
      JOIN ads a ON a.id=f.ad_id
      JOIN campaigns c ON c.id=a.campaign_id
      WHERE c.slug=%s AND a.id = ANY(%s)
      ORDER BY a.id
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug, ad_pks))
        rows = cur.fetchall()

    def _by(v: str) -> List[str]:
        if not v:
            return []
        if v.strip().startswith("["):
            try:
                return [x for x in json.loads(v) if str(x).strip()]
            except Exception:
                return [v]
        return [v]

    out=[]
    for ad_pk, camp_id, slug, fused, page_name, by_raw, media_id in rows:
        out.append({
            "ad_pk": int(ad_pk),
            "campaign_id": camp_id,
            "campaign_slug": slug,
            "fused": fused if isinstance(fused, dict) else json.loads(fused or "{}"),
            "page_name": (page_name or "").strip() or "Unbekannte Gruppe",
            "bylines": _by(by_raw),
            "media_id": str(media_id or "").strip(),
        })
    return out

# ---------------- Analyse & Persistenz ----------------
def tag_ads_to_db(
    conn,
    campaign_slug: str,
    ad_pks: List[int],
    api_key: str,
    model: str = "gpt-4o-mini",
    force: bool = False,
    progress_cb: Optional[Callable[[int,int,str],None]] = None
) -> None:
    """Analysiert ausgewählte Ads (oder skipt vorhandene) und schreibt in ad_topics_results."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    rows = _fetch_fused_rows(conn, campaign_slug, ad_pks)
    n = len(rows)

    def txt_from_fused(fused: Dict[str, Any]) -> str:
        raw = ((fused or {}).get("api") or {}).get("raw", {}) or {}
        parts: List[str] = []
        for k in ("ad_creative_bodies","ad_creative_link_titles","ad_creative_link_descriptions"):
            v = raw.get(k)
            if isinstance(v, list):
                parts.extend([str(x) for x in v])
            elif isinstance(v, str):
                parts.append(v)
        return "\n".join([p for p in parts if p]).strip()

    for i, r in enumerate(rows):
        if progress_cb: progress_cb(i, n, "prüfe")
        if (not force) and _already_done(conn, r["ad_pk"]):
            continue

        text = txt_from_fused(r["fused"])
        meta = {
            "actor": r["page_name"],
            "platforms": ((r["fused"].get("api") or {}).get("raw") or {}).get("publisher_platforms"),
        }

        if progress_cb: progress_cb(i, n, "analysiere")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":
                        f"EINGABE\n- id:{r['ad_pk']}\n- text:{text}\n- meta:{json.dumps(meta, ensure_ascii=False)}\n\nAUSGABE nur JSON."}
                ],
                temperature=0.1
            )
            parsed = _json_first((resp.choices[0].message.content or "").strip())
        except Exception:
            parsed = {}

        topics = parsed.get("topics")
        if not isinstance(topics, list) or not topics:
            topics = ["Sonstiges"]
        conf = parsed.get("confidence")
        try:
            conf = float(conf)
        except Exception:
            conf = None
        bullets = parsed.get("rationale_bullets")
        if not isinstance(bullets, list):
            bullets = []

        if progress_cb: progress_cb(i, n, "speichere")
        _upsert_row(conn, {
            "ad_id": r["ad_pk"],
            "campaign_id": r["campaign_id"],
            "campaign_slug": r["campaign_slug"],
            "page_name": r["page_name"],
            "bylines": r["bylines"],
            "media_id": r["media_id"],
            "topics": json.dumps(topics),
            "rationale_bullets": json.dumps(bullets[:8]),
            "confidence": conf,
            "model": model
        })
        conn.commit()

def fetch_topics_results(conn, campaign_slug: str, ad_pks: List[int]) -> pd.DataFrame:
    # Immer konsistente Spalten bereitstellen
    cols = ["ad_pk", "page_name", "bylines", "topics", "confidence", "rationale_bullets"]
    if not ad_pks:
        return pd.DataFrame(columns=cols)

    sql = """
      SELECT ad_id AS ad_pk, page_name, bylines, topics, confidence, rationale_bullets
      FROM ad_topics_results
      WHERE campaign_slug=%s AND ad_id = ANY(%s)
      ORDER BY analyzed_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug, ad_pks))
        rows = cur.fetchall()

    out = []
    for ad_pk, page_name, bylines, topics, conf, bullets in rows:
        out.append({
            "ad_pk": int(ad_pk),
            "page_name": page_name,
            "bylines": bylines or [],
            "topics": topics if isinstance(topics, list) else (topics and json.loads(topics)) or [],
            "confidence": conf,
            "rationale_bullets": bullets if isinstance(bullets, list) else (bullets and json.loads(bullets)) or []
        })

    return pd.DataFrame(out, columns=cols)

def list_analyzed_ad_ids(conn, campaign_slug: str) -> List[int]:
    """gibt ad_ids zurück, die bereits in ad_topics_results für diese Kampagne stehen"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ad_id FROM ad_topics_results WHERE campaign_slug=%s",
            (campaign_slug,)
        )
        rows = cur.fetchall()
    return [int(r[0]) for r in rows]

__all__ = [
    "ensure_topics_table",
    "list_campaigns",
    "list_fused_ads_for_campaign",
    "tag_ads_to_db",
    "fetch_topics_results",
]
