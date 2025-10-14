# scripts/weaknesses_analyzer.py
from __future__ import annotations
import json
import math
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# --- helpers for new JSON shape ---
def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

_CATS = [
    "factual_accuracy",
    "framing_quality",
    "visual_mislead",
    "targeting_risks",
    "policy_legal",
    "transparency_context",
    "consistency_history",
    "usability_accessibility",
]

def _flatten_record(rec: dict) -> dict:
    """
    Erwartet das LLM-JSON:
      {"id":..., "overall_risk":..., "overall_confidence":..., "categories": {<cat>: {"score":..,"confidence":..,"rationale":..,"examples":[..]} }, "notes": ...}
    und liefert flache Spalten: score_<cat>, confidence_<cat>, rationale_<cat>, examples_<cat>
    """
    out = {
        "id": str(rec.get("id")),
        "overall_risk": _coerce_float(rec.get("overall_risk")),
        "overall_confidence": _coerce_float(rec.get("overall_confidence")),
        "notes": (rec.get("notes") or "").strip(),
    }
    cats = rec.get("categories") or {}
    for key in _CATS:
        c = cats.get(key) or {}
        out[f"score_{key}"] = _coerce_float(c.get("score"))
        out[f"confidence_{key}"] = _coerce_float(c.get("confidence"))
        out[f"rationale_{key}"] = (c.get("rationale") or "").strip()
        ex = c.get("examples")
        out[f"examples_{key}"] = ex if isinstance(ex, list) else []
    return out

# =========================
# System-Prompt
# =========================
DEFAULT_SYSTEM_PROMPT_WEAK = """Du bist Prüfer:in für politische Online-Werbung.

ZIEL  
Analysiere EINE politische Anzeige (Ad-Level) in genau den 8 Kategorien unten.  
Für jede Kategorie sollst du liefern:  
- `score`: Zahl zwischen 0.0 und 1.0 (0.0 = keinerlei Risiko, 1.0 = sehr hohes Risiko; bitte keine 0.0 oder 1.0 exakt)  
- `rationale`: kurze Begründung (max. 25 Wörter)  
- `examples`: Liste von maximal 2 Beispielen oder Claims aus der Anzeige, die zu diesem Risiko passen (oder leer, wenn nichts gefunden)  

AUSGABEFORMAT — nur gültiges JSON (keine Erklärungen, kein Markdown, kein Fließtext außerhalb von JSON):

{
  "id": "<Ad-ID>",
  "overall_risk": 0.0,
  "categories": {
    "factual_accuracy": {
      "score": 0.45,
      "rationale": "Prozentangabe ohne Quelle, veraltete Statistik.",
      "examples": ["'Kriminalität um 50% gesunken' ohne Quelle"]
    },
    "framing_quality": {
      "score": 0.30,
      "rationale": "Emotionsbetontes Framing ohne inhaltliche Belege.",
      "examples": ["'Jetzt handeln, bevor es zu spät ist'"]
    },
    "visual_mislead": {
      "score": 0.60,
      "rationale": "Abgeschnittene Y-Achse im Diagramm.",
      "examples": ["Diagramm zu Arbeitslosigkeit verzerrt dargestellt"]
    },
    "targeting_risks": {
      "score": 0.20,
      "rationale": "Keine klaren Hinweise auf sensitives Targeting.",
      "examples": []
    },
    "policy_legal": {
      "score": 0.55,
      "rationale": "Kein Sponsorenhinweis vorhanden.",
      "examples": []
    },
    "transparency_context": {
      "score": 0.40,
      "rationale": "Versprechen ohne Kostenangaben.",
      "examples": ["'Kostenlos für alle' ohne Finanzierung"]
    },
    "consistency_history": {
      "score": 0.35,
      "rationale": "Abweichung zu früherer Position des Absenders.",
      "examples": []
    },
    "usability_accessibility": {
      "score": 0.25,
      "rationale": "Kontraste niedrig, CTA nicht klar.",
      "examples": []
    }
  },
  "notes": "max. 2 Sätze mit allgemeinen Beobachtungen."
}

KATEGORIEDEFINITIONEN:  
1) `factual_accuracy` – Präzise Behauptungen ohne Quelle; Cherry Picking; Superlative ohne Beleg; falsche Kausalität; veraltete Daten.  
2) `framing_quality` – Logische Fehlschlüsse; extreme Modallogik; Angst-/Empörungs-Framing ohne Substanz; Widersprüche.  
3) `visual_mislead` – Irreführende Visualisierung; Stock-Fotos ohne Kennzeichnung; Deepfake-Indizien; Dark Patterns.  
4) `targeting_risks` – Hypertargeting sensibler Merkmale; Dogwhistles; inkonsistente Botschaften für Zielgruppen.  
5) `policy_legal` – Fehlende Sponsorenangabe; Impersonation; Hate Speech; Urheberrechtsprobleme.  
6) `transparency_context` – Kein Kontext zu Maßnahmen/Kosten; unklare Metriken; selektive Regionendarstellung.  
7) `consistency_history` – Positionswechsel; Wiederholung alter Statistiken ohne Aktualisierung.  
8) `usability_accessibility` – Fehlende Barrierefreiheit; irreführende CTAs.  

REGELN:  
- Immer für jede Kategorie einen Wert >0.0 und <1.0 zurückgeben, auch wenn Risiko gering ist.  
- rationale = max. 25 Wörter, examples max. 2 kurze Einträge.  
- overall_risk = Durchschnitt aller Kategorie-Scores.  
- Keine Werte leer lassen, stattdessen Beispiele-Array leer.  

ERWEITERUNG (KONFIDENZEN):  
Ergänze zusätzlich:
- `overall_confidence` (0.0–1.0) = deine Sicherheit über die Gesamteinschätzung.
- In jeder Kategorie ein Feld `confidence` (0.0–1.0).
Beispiel (Auszug):
{
  "id":"<Ad-ID>",
  "overall_risk": 0.42,
  "overall_confidence": 0.72,
  "categories": {
    "factual_accuracy": {"score":0.45,"confidence":0.70,"rationale":"...","examples":["..."]},
    "framing_quality": {"score":0.30,"confidence":0.65,"rationale":"...","examples":["..."]},
    ...
  },
  "notes":"..."
}
"""

def _coerce_float_bounded(x: Any, lo: float, hi: float, default: Optional[float]) -> Optional[float]:
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        # weich klemmen
        val = max(lo, min(hi, val))
        return round(val, 2)
    except Exception:
        return default

def _safe_json_first(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    # heuristisch: ab erster '{'
    if "{" in s:
        s = s[s.index("{"):]
    return json.loads(s)

# --- leichte Heuristik als Fallback auf KATEGORIEN-Niveau ---
_ATT_WORDS = re.compile(r"\b(skandal|lüge|versagen|korrupt|verrat|schuld|fail|empörung)\b", re.I)
_CTA_WORDS = re.compile(r"\b(jetzt|sofort|teilen|unterstützen|spende[n]?|wählen)\b", re.I)
_NUMERIC   = re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[,\.]\d+)?\b")

def _heuristic_categories(ad_text: str) -> Dict[str, Any]:
    """Erzeugt ein JSON im neuen Kategorien-Format (für _flatten_record)."""
    txt = ad_text or ""
    base = 0.2

    # einfache Indikatoren
    has_num = bool(_NUMERIC.search(txt))
    has_cta = bool(_CTA_WORDS.search(txt))
    has_attack = bool(_ATT_WORDS.search(txt))

    cats = {
        "factual_accuracy": {
            "score": base + (0.15 if has_num else 0.0),
            "confidence": 0.6,
            "rationale": "Zahlen ohne klare Quelle." if has_num else "Keine auffälligen Zahlenangaben.",
            "examples": [],
        },
        "framing_quality": {
            "score": base + (0.15 if has_cta or has_attack else 0.0),
            "confidence": 0.6,
            "rationale": "Emotionaler CTA/Angriffs-Vokabular." if (has_cta or has_attack) else "Sachlicher Ton.",
            "examples": [],
        },
        "visual_mislead": {
            "score": base,
            "confidence": 0.5,
            "rationale": "Keine Visuals-Analyse vorhanden.",
            "examples": [],
        },
        "targeting_risks": {
            "score": base,
            "confidence": 0.5,
            "rationale": "Keine Hinweise auf sensitives Targeting im Text.",
            "examples": [],
        },
        "policy_legal": {
            "score": base,
            "confidence": 0.5,
            "rationale": "Policy-Aspekte im Text nicht belegt.",
            "examples": [],
        },
        "transparency_context": {
            "score": base + (0.1 if has_num else 0.0),
            "confidence": 0.6,
            "rationale": "Kontext/Belege zu Zahlen unklar." if has_num else "Kontextlage unauffällig.",
            "examples": [],
        },
        "consistency_history": {
            "score": base,
            "confidence": 0.5,
            "rationale": "Keine Historie im Text erkennbar.",
            "examples": [],
        },
        "usability_accessibility": {
            "score": base,
            "confidence": 0.5,
            "rationale": "Ohne Creative-Details schwer bewertbar.",
            "examples": [],
        },
    }

    overall = round(sum(v["score"] for v in cats.values()) / len(cats), 2)

    return {
        "id": "",
        "overall_risk": overall,
        "overall_confidence": 0.6,
        "categories": cats,
        "notes": "",
    }

def analyze_weaknesses(df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> pd.DataFrame:
    """
    Erwartet df mit Spalten: id, ad_text, platforms, start_date, end_date, creative_features (optional)
    Gibt DataFrame mit Spalten:
      id, overall_risk, overall_confidence,
      score_<cat>, confidence_<cat>, rationale_<cat>, examples_<cat> (für alle 8 Kategorien),
      notes
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except Exception:
        # Kein Client -> Heuristik-only
        rows = []
        for _, r in df.iterrows():
            heur = _heuristic_categories(r.get("ad_text") or "")
            heur["id"] = str(r.get("id") or "")
            rows.append(_flatten_record(heur))
        return _order_columns(pd.DataFrame(rows))

    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ad_id = str(row.get("id") or "")
        # kompakter User-Prompt (inkl. Creative-Keys als Hinweis)
        meta = {
            "platforms": row.get("platforms"),
            "period": [str(row.get("start_date")), str(row.get("end_date"))],
        }
        cf = row.get("creative_features") or {}
        if isinstance(cf, dict) and cf:
            meta["creative_features"] = {
                "vis": list((cf.get("visuelle_features") or {}).keys()),
                "txt": list((cf.get("textuelle_features") or {}).keys()),
                "sem": list((cf.get("semantische_features") or {}).keys()),
            }
        user_prompt = (
            "Analysiere diese politische Anzeige und liefere JSON nach dem vorgegebenen Schema.\n"
            f"id: {ad_id}\n"
            f"text: {row.get('ad_text') or ''}\n"
            f"meta: {json.dumps(meta, ensure_ascii=False)}"
        )

        parsed: Dict[str, Any] = {}
        # bis zu 2 Versuche (Parsing/Guardrails)
        for _attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_WEAK},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.15,
                    max_tokens=900,
                    n=1,
                )
                content = (resp.choices[0].message.content or "").strip()
                parsed = _safe_json_first(content)
                break
            except Exception:
                parsed = {}

        if not isinstance(parsed, dict) or not parsed.get("categories"):
            # Heuristik-Fallback in NEUEM Format
            heur = _heuristic_categories(row.get("ad_text") or "")
            heur["id"] = ad_id
            out_rows.append(_flatten_record(heur))
            continue

        # Minimal normalisieren/absichern
        if "id" not in parsed or not parsed["id"]:
            parsed["id"] = ad_id

        # Soft-Klemmung für overall_risk/overall_confidence
        parsed["overall_risk"] = _coerce_float_bounded(parsed.get("overall_risk"), 0.05, 0.95, 0.25)
        parsed["overall_confidence"] = _coerce_float_bounded(parsed.get("overall_confidence"), 0.0, 1.0, 0.7)

        # Kategorien: fehlende Scores/Conf auffüllen (und weich klemmen)
        cats = parsed.get("categories") or {}
        for k in _CATS:
            c = cats.get(k) or {}
            sc  = _coerce_float_bounded(c.get("score"), 0.05, 0.95, 0.2)
            cof = _coerce_float_bounded(c.get("confidence"), 0.0, 1.0, 0.7)
            rat = (c.get("rationale") or "").strip()
            ex  = c.get("examples") if isinstance(c.get("examples"), list) else []
            cats[k] = {"score": sc, "confidence": cof, "rationale": rat, "examples": ex[:2]}
        parsed["categories"] = cats

        out_rows.append(_flatten_record(parsed))

    return _order_columns(pd.DataFrame(out_rows))


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Einheitliche, gut lesbare Spaltenreihenfolge."""
    ordered = (
        ["id", "overall_risk", "overall_confidence"]
        + sum(([f"score_{k}", f"confidence_{k}"] for k in _CATS), [])
        + [f"rationale_{k}" for k in _CATS]
        + [f"examples_{k}" for k in _CATS]
        + ["notes"]
    )
    cols = [c for c in ordered if c in df.columns]
    # ggf. Rest anhängen
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]
