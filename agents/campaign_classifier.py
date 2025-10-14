# agents/campaign_classifier.py
import json
from typing import Any, Dict, List, Optional, Callable
import pandas as pd

from agents._db import connect
from agents.ad_tagger import list_campaigns, list_fused_ads_for_campaign

SYSTEM_PROMPT = """Bewerte die Haltung eines Akteurs zu EINEM Thema innerhalb EINER Kampagne.

KONTEXT
- Grundlage sind mehrere Anzeigen (Bullets = verdichtete Hinweise aus Ad-Analysen).
- Ziel ist eine robuste Einschätzung der Perspektive (Pro/Contra/Neutral/Unklar).

DEFINITIONEN
- stance:
  • Pro     = fördert/befürwortet Maßnahme/Thema
  • Contra  = lehnt Maßnahme/Thema ab / bekämpft sie
  • Neutral = informiert/bilanziert ohne klare Richtung
  • Unklar  = zu wenig/kontradiktorische Hinweise
- confidence (0.0–1.0): 0.3 schwach, 0.6 solide Mehrheitssignale, 0.85 sehr konsistent.

ANTWORTFORMAT (nur JSON):
{
  "actor": "<Gruppe>",
  "topic": "<Thema (aus Ontologie des Ad-Taggers)>",
  "stance": "Pro|Contra|Neutral|Unklar",
  "confidence": 0.76,
  "rationale_bullets": ["<kurz>","<kurz>","<kurz>"]
}
"""

def ensure_perspective_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS campaign_perspective_results (
            id                BIGSERIAL PRIMARY KEY,
            campaign_slug     TEXT NOT NULL,
            page_name         TEXT NOT NULL,
            topic             TEXT NOT NULL,
            stance            TEXT NOT NULL,
            confidence        DOUBLE PRECISION,
            rationale_bullets JSONB,
            model             TEXT,
            ad_ids            JSONB,
            analyzed_at       TIMESTAMPTZ DEFAULT now(),
            UNIQUE (campaign_slug, page_name, topic)
        )""")
    conn.commit()

def _fetch_topics_for_ads(conn, campaign_slug: str, ad_pks: List[int]) -> pd.DataFrame:
    if not ad_pks:
        return pd.DataFrame()
    sql = """
      SELECT t.ad_id AS ad_pk, t.page_name, t.topics, t.rationale_bullets
      FROM ad_topics_results t
      WHERE t.campaign_slug=%s AND t.ad_id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug, ad_pks))
        rows = cur.fetchall()
    out=[]
    for ad_pk, page_name, topics, bullets in rows:
        out.append({
            "ad_pk": int(ad_pk),
            "page_name": page_name,
            "topics": topics if isinstance(topics, list) else (topics and json.loads(topics)) or [],
            "bullets": bullets if isinstance(bullets, list) else (bullets and json.loads(bullets)) or [],
        })
    return pd.DataFrame(out)

def _already_done(conn, campaign_slug: str, actor: str, topic: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""SELECT 1 FROM campaign_perspective_results
                       WHERE campaign_slug=%s AND page_name=%s AND topic=%s LIMIT 1""",
                    (campaign_slug, actor, topic))
        return cur.fetchone() is not None

def _upsert(conn, row: Dict[str, Any]) -> None:
    sql = """
    INSERT INTO campaign_perspective_results
      (campaign_slug,page_name,topic,stance,confidence,rationale_bullets,model,ad_ids,analyzed_at)
    VALUES
      (%(campaign_slug)s,%(page_name)s,%(topic)s,%(stance)s,%(confidence)s,%(rationale_bullets)s,%(model)s,%(ad_ids)s,now())
    ON CONFLICT (campaign_slug,page_name,topic) DO UPDATE SET
      stance=EXCLUDED.stance,
      confidence=EXCLUDED.confidence,
      rationale_bullets=EXCLUDED.rationale_bullets,
      model=EXCLUDED.model,
      ad_ids=EXCLUDED.ad_ids,
      analyzed_at=now()
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)

def classify_perspective_to_db(
    conn,
    campaign_slug: str,
    ad_pks: List[int],
    api_key: str,
    model: str = "gpt-4o-mini",
    force: bool = False,
    progress_cb: Optional[Callable[[int,int,str],None]] = None
) -> None:
    """Gruppiert Actor×Topic aus vorhandenen ad_topics_results und klassifiziert deren Haltung."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    topics_df = _fetch_topics_for_ads(conn, campaign_slug, ad_pks)
    if topics_df.empty:
        return

    rows=[]
    for _, r in topics_df.iterrows():
        actor = r["page_name"] or "Unbekannt"
        for t in (r["topics"] or ["Sonstiges"]):
            rows.append({"actor": actor, "topic": t, "ad_pk": r["ad_pk"], "bullets": r["bullets"]})
    grp = pd.DataFrame(rows)
    if grp.empty:
        return

    groups = grp.groupby(["actor","topic"])
    n = len(groups)

    for i, ((actor, topic), g) in enumerate(groups):
        if progress_cb: progress_cb(i, n, "prüfe")
        if (not force) and _already_done(conn, campaign_slug, actor, topic):
            continue

        bullets=[]
        for bl in g["bullets"].tolist():
            if isinstance(bl, list):
                bullets.extend([str(x) for x in bl])
        bullets = bullets[:8]

        user = f"Akteur: {actor}\nThema: {topic}\nStichpunkte:\n- " + "\n- ".join(bullets or ["(keine)"])

        if progress_cb: progress_cb(i, n, "analysiere")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":user}],
                temperature=0.1
            )
            parsed = resp.choices[0].message.content or ""
            obj = json.loads(parsed[parsed.find("{"):]) if "{" in parsed else {}
        except Exception:
            obj = {}

        stance = obj.get("stance") if obj.get("stance") in ("Pro","Contra","Neutral","Unklar") else "Unklar"
        try:
            conf = float(obj.get("confidence"))
        except Exception:
            conf = None
        rb = obj.get("rationale_bullets") if isinstance(obj.get("rationale_bullets"), list) else []

        if progress_cb: progress_cb(i, n, "speichere")
        _upsert(conn, {
            "campaign_slug": campaign_slug,
            "page_name": actor,
            "topic": topic,
            "stance": stance,
            "confidence": conf,
            "rationale_bullets": json.dumps(rb[:8]),
            "model": model,
            "ad_ids": json.dumps(sorted(set(g["ad_pk"].tolist())))
        })
        conn.commit()

def fetch_perspective_results(conn, campaign_slug: str) -> pd.DataFrame:
    # Immer konsistente Spalten – auch wenn keine Zeilen vorhanden sind
    cols = ["page_name", "topic", "stance", "confidence", "rationale_bullets", "analyzed_at"]
    sql = """
      SELECT page_name, topic, stance, confidence, rationale_bullets, analyzed_at
      FROM campaign_perspective_results
      WHERE campaign_slug=%s
      ORDER BY page_name, topic
    """
    with conn.cursor() as cur:
        cur.execute(sql, (campaign_slug,))
        rows = cur.fetchall()

    out = []
    for page_name, topic, stance, conf, rb, analyzed_at in rows:
        out.append({
            "page_name": page_name,
            "topic": topic,
            "stance": stance,
            "confidence": conf,
            "rationale_bullets": rb if isinstance(rb, list) else (rb and json.loads(rb)) or [],
            "analyzed_at": analyzed_at,
        })

    return pd.DataFrame(out, columns=cols)

__all__ = [
    "ensure_perspective_table",
    "classify_perspective_to_db",
    "fetch_perspective_results",
    "list_campaigns",
    "list_fused_ads_for_campaign",
]
