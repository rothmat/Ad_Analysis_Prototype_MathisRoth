# Politische Werbung – Analyse-Dashboard (Streamlit Prototype)
# -----------------------------------------------------------
import json
from datetime import date, timedelta
from dateutil.parser import isoparse
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import os, json
from agents.ad_tagger import tag_ads
from agents.weaknesses_analyzer import analyze_weaknesses

# --- Feature-Flag: unterstützt diese Streamlit-Version ButtonColumn?
try:
    SUPPORTS_BUTTONCOL = hasattr(st, "column_config") and hasattr(st.column_config, "ButtonColumn")
except Exception:
    SUPPORTS_BUTTONCOL = False

st.set_page_config(page_title="Politische Werbung – Analyse", layout="wide")
if "show_ad_table" not in st.session_state:
    st.session_state["show_ad_table"] = False
if "show_camp_table" not in st.session_state:
    st.session_state["show_camp_table"] = False

# ---- UI-Feature-Detection & Overlay-Helfer ----
SUPPORTS_BUTTONCOL = hasattr(st, "column_config") and hasattr(st.column_config, "ButtonColumn")
SUPPORTS_MODAL     = hasattr(st, "dialog")  # neuere Streamlit-Versionen

# ⚠️ Umbenannt: reine Trace-Ansicht (ohne inhaltliche Begründung)
def _show_ad_llm_traces_overlay(ad_id: str):
    """Zeigt die gespeicherten LLM-Traces (Prompts/Usage/Raw) für eine Ad-ID."""
    params = (st.session_state.get("ad_tagging_params") or {}).get(str(ad_id))
    raw    = (st.session_state.get("ad_tagging_raw") or {}).get(str(ad_id))
    usage  = (st.session_state.get("ad_tagging_usage") or {}).get(str(ad_id))

    def _render():
        st.markdown(f"### 🔍 Details zu Ad {ad_id}")
        if params:
            st.subheader("Verwendete Parameter")
            st.json(params)
        if usage:
            st.subheader("Token/Usage")
            st.json(usage)
        if raw:
            st.subheader("Rohantwort")
            st.json(raw)
        if not any([params, usage, raw]):
            st.info("Keine Trace-Daten gespeichert.")

    if SUPPORTS_MODAL and hasattr(st, "dialog"):
        with st.dialog("LLM-Details"):
            _render()
            st.button("Schließen")
    else:
        with st.expander(f"🔍 LLM-Details zu Ad {ad_id}", expanded=True):
            _render()

# -----------------------------
# Helpers
# -----------------------------
def safe_date(s):
    try:
        return isoparse(str(s)).date()
    except Exception:
        return None

def midpoint(lo, hi):
    def f(x):
        try:
            return float(x)
        except Exception:
            return None
    lo, hi = f(lo), f(hi)
    if lo is None and hi is None: return None
    if lo is None: return hi
    if hi is None: return lo
    return (lo + hi)/2.0

TOPIC_KEYWORDS = {
    "Klima & Energie": ["klima","klimaschutz","co2","energie","solar","photovoltaik","wind","erneuerbar"],
    "Migration & Integration": ["migration","migrant","flücht","asyl","integration","grenze","abschieb"],
    "Soziales & Verteilung": ["sozial","rente","löhne","miete","lebenshaltung","armut","gerecht","bürgergeld"],
    "Mobilität & ÖV": ["öpnv","öpv","ticket","bahn","takt","pendler"],
    "Sicherheit & Ordnung": ["polizei","kriminal","innere sicherheit","gewalt","terror"],
    "Außen & Entwicklung": ["afrika","eu","ukraine","nato","entwicklungshilfe","ausland"],
    "Wirtschaft & Innovation": ["wirtschaft","innovation","arbeitsplätze","jobs","industrie","mittelstand"],
}

def heuristic_topics(text: str):
    t = (text or "").lower()
    hits = [k for k, kws in TOPIC_KEYWORDS.items() if any(kw in t for kw in kws)]
    return hits[:3] if hits else ["Sonstiges"]

def heuristic_strategy(row: pd.Series):
    strategy = []
    # Hypertargeting-Proxy
    audience_div = len(row.get("age_gender_breakdowns") or [])
    regions_n = int(row.get("regions_count") or 0)
    if audience_div >= 5 and regions_n >= 8:
        strategy.append("Hypertargeting")
    txt = f"{row.get('ad_text','')}".lower()
    if any(w in txt for w in ["jetzt","unterstützen","wählen","stimmen","teilen"]):
        strategy.append("Mobilisierung")
    if any(w in txt for w in ["fair","kosten","arbeit","jobs","innovation","leistung"]):
        strategy.append("Persuasion")
    return strategy or ["Unklar"]

def expand_meta_record(rec: Dict[str, Any]):
    # Combine creative text fields
    text_pieces = []
    for k in ["ad_creative_bodies","ad_creative_link_titles","ad_creative_link_descriptions"]:
        v = rec.get(k)
        if isinstance(v, list): 
            text_pieces.extend([str(x) for x in v])
        elif isinstance(v, str): 
            text_pieces.append(v)
    ad_text = "\n".join(text_pieces).strip()

    start = safe_date(rec.get("ad_delivery_start_time") or rec.get("ad_creation_time"))
    end = safe_date(rec.get("ad_delivery_stop_time")) or (start or date.today())
    if start and end and end < start:
        end = start

    impressions = midpoint(rec.get("impressions",{}).get("lower_bound"),
                           rec.get("impressions",{}).get("upper_bound"))
    spend = midpoint(rec.get("spend",{}).get("lower_bound"),
                     rec.get("spend",{}).get("upper_bound"))

    regions = rec.get("delivery_by_region") or []
    age_country_gender = rec.get("age_country_gender_reach_breakdown") or []
    breakdown = age_country_gender[0]["age_gender_breakdowns"] if age_country_gender else []

    # --- Creative-Features robust einsammeln (creative_features ODER additional_data) ---
    def _get_creative(rec_: Dict[str, Any]):
        # 1) klassisches Bündel
        if isinstance(rec_.get("creative_features"), dict):
            cf = rec_["creative_features"]
            return {
                "visuelle_features": cf.get("visuelle_features", {}) or {},
                "textuelle_features": cf.get("textuelle_features", {}) or {},
                "semantische_features": cf.get("semantische_features", {}) or {},
            }
        # 2) neues Schema: additional_data.{visuelle_, textuelle_, semantische_}
        if isinstance(rec_.get("additional_data"), dict):
            add = rec_["additional_data"]
            return {
                "visuelle_features": add.get("visuelle_features", {}) or {},
                "textuelle_features": add.get("textuelle_features", {}) or {},
                "semantische_features": add.get("semantische_features", {}) or {},
            }
        # 3) Fallback: evtl. top-level Keys vorhanden
        return {
            "visuelle_features": rec_.get("visuelle_features", {}) or {},
            "textuelle_features": rec_.get("textuelle_features", {}) or {},
            "semantische_features": rec_.get("semantische_features", {}) or {},
        }

    cf = _get_creative(rec)
    if not any(cf.values()):  # wenn alles leer, dann None, damit der Tab sauber skippt
        cf = None

    row = {
        "id": rec.get("id"),
        "advertiser": rec.get("page_name") or rec.get("bylines") or "Unbekannt",
        "campaign": (rec.get("ad_creative_link_titles") or ["Unbekannte Kampagne"])[0],
        "platforms": ", ".join(rec.get("publisher_platforms") or []),
        "ad_text": ad_text,
        "start_date": start,
        "end_date": end,
        "spend_mid": spend,
        "impressions_mid": impressions,
        "currency": rec.get("currency") or "EUR",
        "regions_count": len(regions),
        "regions_json": regions,
        "gender_age_json": rec.get("demographic_distribution") or [],
        "age_gender_breakdowns": breakdown,
        "snapshot_url": rec.get("ad_snapshot_url"),
        "creative_features": cf,  # <— jetzt auch aus additional_data befüllt
    }
    return row

def to_daily(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        sd, ed = r["start_date"], r["end_date"]
        if not sd or not ed: continue
        days = max(1, (ed - sd).days + 1)
        spend_per = (r["spend_mid"] or 0)/days
        imp_per = (r["impressions_mid"] or 0)/days
        for i in range(days):
            d = sd + timedelta(days=i)
            rows.append({"date": d, "id": r["id"], "campaign": r["campaign"],
                         "spend": spend_per, "impressions": imp_per})
    return pd.DataFrame(rows)

def kl_div(p, q, eps=1e-9):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return float(np.sum(p*np.log(p/q)))

def as_pct(x):
    try: return float(x)
    except: return np.nan

# ✅ Dies ist die **sichtbare** Ad-Details-Ansicht – jetzt mit Begründungs-Bullets
def _show_ad_details_overlay(ad_id: str):
    st.session_state.setdefault("ad_tagging_raw", {})
    st.session_state.setdefault("ad_tagging_usage", {})
    st.session_state.setdefault("ad_tagging_params", {})

    # Zeile aus Ergebnissen ziehen
    ad_df = st.session_state.get("ad_tagging_results")
    row = None
    if isinstance(ad_df, pd.DataFrame) and not ad_df.empty:
        m = ad_df[ad_df["id"].astype(str) == str(ad_id)]
        if not m.empty:
            row = m.iloc[0]

    with st.expander(f"🔍 Details – Ad {ad_id}", expanded=True):
        if row is None:
            st.info("Keine kompakten Ergebnisdaten gefunden.")
        else:
            topics   = row.get("topics") or []
            strategy = row.get("strategy") or []
            conf     = row.get("confidence")
            bullets  = row.get("rationale_bullets") or []

            if topics:   st.write("**Themen:** " + ", ".join(map(str, topics)))
            if strategy: st.write("**Strategie:** " + ", ".join(map(str, strategy)))
            if pd.notna(conf): st.write(f"**Confidence:** {float(conf):.2f}")

            # 👇 Neuer Abschnitt – analog zur Kampagnen-Ansicht
            st.markdown("**Begründung (vom Modell):**")
            if isinstance(bullets, list) and bullets:
                for b in bullets[:8]:
                    st.markdown(f"- {b}")
            else:
                st.caption("Keine rationale_bullets im Modell-Output enthalten.")

        with st.expander("⚙ Debug-Details (optional)"):
            params = (st.session_state.get("ad_tagging_params") or {}).get(str(ad_id))
            raw    = (st.session_state.get("ad_tagging_raw") or {}).get(str(ad_id))
            usage  = (st.session_state.get("ad_tagging_usage") or {}).get(str(ad_id))
            if params: st.json({"params": params})
            if usage:  st.json({"usage": usage})
            if raw:    st.json({"raw_response": raw})

def _show_campaign_details_overlay(campaign_id: str):
    """Kompaktes Overlay: zeigt nur Kernfelder + rationale_bullets.
       Debug-Infos (Prompts/Raw/Usage) optional im Expander.
    """
    # — Ergebniszeile aus der zuletzt angezeigten Kampagnen-Tabelle ziehen
    camp_df = st.session_state.get("campaign_results")
    row = None
    if isinstance(camp_df, pd.DataFrame) and not camp_df.empty:
        # Spalten-Varianz robust handlen
        col_id = "campaign_auto_id" if "campaign_auto_id" in camp_df.columns else ("campaign_id" if "campaign_id" in camp_df.columns else None)
        if col_id:
            rowm = camp_df[camp_df[col_id].astype(str) == str(campaign_id)]
            if not rowm.empty:
                row = rowm.iloc[0]

    # Fallback: ohne Row trotzdem ein schlankes Overlay zeigen
    title = f"🔍 Details – Kampagne {campaign_id}"
    with st.expander(title, expanded=True):
        if row is None:
            st.info("Keine kompakten Ergebnisdaten gefunden.")
        else:
            name_guess = row.get("name_guess") or "-"
            topics     = row.get("primary_topics") or []
            strategies = row.get("strategy_mix") or []
            tgt_style  = row.get("targeting_style") or "-"
            objective  = row.get("objective_guess") or "-"
            conf       = row.get("confidence")
            bullets    = row.get("rationale_bullets") or []

            st.markdown(f"### {name_guess}")
            st.write(f"**Kampagnen-ID:** `{campaign_id}`")
            if topics:     st.write("**Primäre Themen:** " + ", ".join(map(str, topics)))
            if strategies: st.write("**Strategie:** " + ", ".join(map(str, strategies)))
            st.write(f"**Targeting:** {tgt_style}  |  **Objective:** {objective}")
            if pd.notna(conf):
                st.write(f"**Confidence:** {float(conf):.2f}")

            st.markdown("**Begründung (vom Modell):**")
            if bullets:
                for b in bullets:
                    st.markdown(f"- {b}")
            else:
                st.caption("Keine rationale_bullets im Modell-Output enthalten.")

        # — Optionaler Debug-Bereich (kompakt versteckt)
        with st.expander("⚙ Debug-Details (optional)"):
            params = (st.session_state.get("campaign_params") or {}).get(str(campaign_id))
            raw    = (st.session_state.get("campaign_raw") or {}).get(str(campaign_id))
            usage  = (st.session_state.get("campaign_usage") or {}).get(str(campaign_id))
            if params: st.json({"params": params})
            if usage:  st.json({"usage": usage})
            if raw:    st.json({"raw_response": raw})
            if not any([params, raw, usage]):
                st.caption("Keine Debug-Metadaten gespeichert.")

# -----------------------------
# Sidebar – Ingest (Multi-File Upload)
# -----------------------------
st.sidebar.title("Daten laden")
ups = st.sidebar.file_uploader(
    "JSON/JSONL/CSV hochladen (mehrere Dateien möglich)",
    type=["json", "jsonl", "csv"],
    accept_multiple_files=True
)
remove_dupes = st.sidebar.checkbox("Duplikate (nach id) entfernen", value=True)
use_demo = st.sidebar.checkbox("Demo-Datensatz laden", value=not ups)

records: List[Dict[str, Any]] = []

if ups:
    for up in ups:
        name = (up.name or "").lower()
        if name.endswith(".jsonl"):
            # Eine Anzeige pro Zeile
            for line in up:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                if line.strip():
                    records.append(json.loads(line))
        elif name.endswith(".json"):
            obj = json.load(up)
            # Sowohl [ {…}, {…} ] als auch {…} werden unterstützt
            records.extend(obj if isinstance(obj, list) else [obj])
        elif name.endswith(".csv"):
            df_csv = pd.read_csv(up)
            # Falls Roh-JSONs in einer "raw"-Spalte liegen
            if "raw" in df_csv.columns:
                records.extend(
                    df_csv["raw"].dropna().map(lambda x: json.loads(x) if isinstance(x, str) else x).tolist()
                )
            else:
                # Bereits flache Spalten – weiter unten vom Mapper verarbeitet
                records.extend(df_csv.to_dict(orient="records"))
elif use_demo:
    # Minimal Demo
    demo = {
      "id": "746419645011412",
      "ad_creation_time": "2025-07-10",
      "ad_delivery_start_time": "2025-07-10",
      "ad_creative_bodies": [
        "Skandal: Deutschland zahlt Menstruationsprogramme in Afrika! Während in Deutschland ... weltfremd!"
      ],
      "ad_creative_link_titles": ["Skandal: Deutschland zahlt Menstruationsprogramme in Afrika!"],
      "ad_snapshot_url": "https://www.facebook.com/ads/archive/render_ad/?id=746419645011412",
      "currency": "EUR",
      "demographic_distribution": [
        {"percentage":"0.301933","age":"65+","gender":"female"},
        {"percentage":"0.245982","age":"65+","gender":"male"},
        {"percentage":"0.187792","age":"55-64","gender":"male"},
        {"percentage":"0.165005","age":"55-64","gender":"female"}
      ],
      "delivery_by_region": [
        {"percentage":"0.205739","region":"Nordrhein-Westfalen"},
        {"percentage":"0.125153","region":"Bayern"},
        {"percentage":"0.10114","region":"Baden-Württemberg"}
      ],
      "impressions":{"lower_bound":"5000","upper_bound":"5999"},
      "page_name": "Dirk Brandes - Für Niedersachsen im Bundestag",
      "publisher_platforms": ["facebook","instagram"],
      "spend":{"lower_bound":"0","upper_bound":"99"},
      "creative_features": {
        "visuelle_features": {"farbpalette":["#00A651","#FFFFFF","#FF00FF"],"textausrichtung":"linksbündig","flächenverteilung":{"textfläche":40,"bildfläche":50,"weißraum":10},"plattform":"Facebook"},
        "textuelle_features": {"cta_typ":"Imperativ","cta_position":"unten","cta_visuelle_prominenz":"mittel","headline_wortanzahl":8},
        "semantische_features": {"argumenttyp":"rational","emotionaler_apell":"Dringlichkeit","framing_typ":"Moralisch"}
      }
    }
    records = [demo]

# Optional: Duplikate entfernen (nach 'id')
if remove_dupes and records:
    seen = set()
    deduped = []
    for r in records:
        rid = r.get("id")
        if rid not in seen:
            seen.add(rid)
            deduped.append(r)
    records = deduped

if not records:
    st.stop()

# Mapping ins interne Schema + DataFrame
rows = [expand_meta_record(r) for r in records]
df = pd.DataFrame(rows)

# creative-Spalte global absichern (für spätere Kampagnen-Klassifizierung)
if "creative" not in df.columns and "creative_features" in df.columns:
    df["creative"] = df["creative_features"]
if "creative" not in df.columns:
    df["creative"] = [{} for _ in range(len(df))]

# =============================
# LLM-Trigger (UI) + Live-Ergebnisse
# =============================
st.sidebar.header("LLM-Trigger")
with st.sidebar.expander("Einstellungen", expanded=True):
    api_key_ui = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    model_ui   = st.selectbox("Modell", ["gpt-4o-mini","gpt-4o","gpt-4.1"], index=0)
    run_ads    = st.button("🔎 Ad-Tagging: Themen + Strategie (LLM) jetzt starten")
    run_camps  = st.button("🧭 Kampagnen-Klassifizierung (LLM) jetzt starten")
    # --- Button in der Sidebar (analog Ad-Tagging / Kampagnen) ---
    st.sidebar.header("Schwachstellen-Analyse")
    run_weak = st.sidebar.button("🛡️ Schwachstellen (LLM) jetzt starten")

def _hash_text(s: str):
    import hashlib
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

DEFAULT_SYSTEM_PROMPT = """Du bist ein Analyst für politische Online-Werbung. 
Ziele:
- Klassifiziere Anzeigentexte in vordefinierte Themen (max. 3).
- Bestimme die Kommunikationsstrategie(n).
- Antworte ausschließlich mit gültigem kompaktem JSON gemäß Schema, ohne Fließtext.

Themenkategorien (exakt so verwenden):
["Klima & Energie","Mobilität & ÖV","Soziales & Verteilung","Migration & Integration","Sicherheit & Ordnung","Wirtschaft & Innovation","Außen & Entwicklung","Sonstiges"]

Strategie-Tags (exakt so verwenden):
["Hypertargeting","Mobilisierung","Persuasion","Kontrastierung"]

Regeln:
- Max. 3 Themen, keine Synonyme/Varianten.
- Wenn unklar: nutze "Sonstiges" (Themen) bzw. ["Unklar"] (Strategie).
- Gib nur JSON zurück; kein Markdown, keine Erklärungen, keine Codefences.
- Schema:
  {"id":"<ID>","topics":["<Thema1>","<Thema2>"],"strategy":["<Strategie1>"],"confidence":0.0}
"""

def _user_prompt_for_ad(ad_id, text, meta):
    return (
        f"Input:\n"
        f'- id: {ad_id}\n'
        f'- text: {text}\n'
        f'- meta: {json.dumps(meta, ensure_ascii=False)}\n\n'
        'Ausgabeschema (nur JSON ausgeben): '
        f'{{"id":"{ad_id}","topics":["<Thema1>"],"strategy":["<Strategie>"],"confidence":0.0}}'
    )

def _safe_json_first(s: str) -> Dict[str, Any]:
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

def _llm_tag_ads(df_subset: pd.DataFrame, api_key: str, model: str) -> pd.DataFrame:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAI-Client nicht verfügbar: {e}")
        return pd.DataFrame([])

    cache, rows = {}, []
    for _, r in df_subset.iterrows():
        ad_id = str(r["id"])
        text = r.get("ad_text") or ""
        key = (ad_id, _hash_text(text))
        if key in cache:
            rows.append({"id": ad_id, **cache[key]})
            continue

        meta = {
            "platforms": r.get("platforms"),
            "period": [str(r.get("start_date")), str(r.get("end_date"))],
            "regions_n": r.get("regions_count"),
        }
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user",   "content": _user_prompt_for_ad(ad_id, text, meta)},
                ],
                temperature=0.1,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _safe_json_first(raw)
        except Exception:
            parsed = {}

        topics   = parsed.get("topics")   if isinstance(parsed.get("topics"), list)   and parsed.get("topics")   else ["Sonstiges"]
        strategy = parsed.get("strategy") if isinstance(parsed.get("strategy"), list) and parsed.get("strategy") else ["Unklar"]
        conf     = parsed.get("confidence") if isinstance(parsed.get("confidence"), (int,float)) else None
        payload  = {"topics": topics, "strategy": strategy, "confidence": conf}
        cache[key] = payload
        rows.append({"id": ad_id, **payload})
    return pd.DataFrame(rows)

# Kampagnen-Funktionen (aus scripts/)
try:
    from agents.campaign_classifier import detect_campaigns, classify_campaigns
    _campaigns_available = True
except Exception as e:
    _campaigns_available = False
    st.warning(f"Kampagnenfunktionen nicht importierbar (scripts/campaign_classifier.py). Grund: {e}")

# ---------- Pretty helpers ----------
def _fmt_lists_for_table(df: pd.DataFrame, list_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in list_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else v)
    return out

def _show_named_table(df: pd.DataFrame, rename_map: dict, list_cols: list[str], title: str):
    if df is None or df.empty:
        st.caption(f"Keine Daten für: {title}")
        return
    pretty = _fmt_lists_for_table(df, list_cols)
    pretty = pretty.rename(columns=rename_map)
    st.subheader(title)
    st.dataframe(pretty, use_container_width=True)

def _open_llm_details_modal(title: str, payload: dict):
    """Zeigt ein Modal mit hübsch formatiertem JSON."""
    if hasattr(st, "modal"):
        with st.modal(title):
            st.json(payload)  # pretty JSON
            st.caption("Hinweis: Falls `raw_response`/`tokens` leer sind, liefert das jeweilige Script diese Meta-Daten aktuell nicht zurück.")

def _build_ad_tagging_trace_row(row: pd.Series) -> dict:
    """Best-Effort LLM-Trace für Ad-Tagging (Input/Output/Parameter).
       Nutzt vorhandene UI-Infos + Session State. """
    ad_id = str(row.get("id") or "")
    # was wir sicher wissen
    meta = {
        "platforms": row.get("platforms"),
        "period": [str(row.get("start_date")), str(row.get("end_date"))],
        "regions_n": row.get("regions_count"),
    }
    trace = {
        "kind": "ad_tagging",
        "id": ad_id,
        "model": st.session_state.get("last_model_used", None) or "unknown",
        "temperature": 0.1,  # so verwenden wir es in _llm_tag_ads; falls dein scripts/ad_tagger abweicht, dort nachziehen
        "system_prompt": st.session_state.get("ad_tagging_system_prompt"),  # s. Patch unten
        "user_meta": meta,
        "input_excerpt": (row.get("ad_text") or "")[:800],
        "output": {
            "topics": row.get("topics"),
            "strategy": row.get("strategy"),
            "confidence": row.get("confidence"),
        },
        "raw_response": st.session_state.get("ad_tagging_raw", {}).get(ad_id),  # optional, s. Patch unten
        "tokens": st.session_state.get("ad_tagging_usage", {}).get(ad_id),      # optional
    }
    return trace

def _build_campaign_trace_row(row: pd.Series) -> dict:
    """Best-Effort LLM-Trace für Kampagnenklassifizierung."""
    cid = str(row.get("campaign_auto_id") or row.get("campaign_id") or "")
    trace = {
        "kind": "campaign_classification",
        "campaign_id": cid,
        "model": st.session_state.get("last_model_used", None) or "unknown",
        "temperature": 0.15,  # so nutzen wir es i.d.R. für Analyse-Modelle
        "system_prompt": st.session_state.get("campaign_system_prompt"),
        "input_ads": row.get("ad_ids"),
        "output": {
            "name_guess": row.get("name_guess"),
            "primary_topics": row.get("primary_topics"),
            "strategy_mix": row.get("strategy_mix"),
            "targeting_style": row.get("targeting_style"),
            "objective_guess": row.get("objective_guess"),
            "confidence": row.get("confidence"),
            "rationale_bullets": row.get("rationale_bullets"),
        },
        "raw_response": st.session_state.get("campaign_raw", {}).get(cid),   # optional
        "tokens": st.session_state.get("campaign_usage", {}).get(cid),       # optional
    }
    return trace

# ---- Ad-Tagging: Ergebnis NUR speichern, Anzeige unten zentral ----
if run_ads:
    if not api_key_ui:
        st.error("Bitte OpenAI API Key angeben.")
    else:
        st.session_state["last_model_used"] = model_ui
        st.session_state["ad_tagging_system_prompt"] = DEFAULT_SYSTEM_PROMPT
        # optional container, falls Scripts raw usage liefern:
        st.session_state.setdefault("ad_tagging_raw", {})
        st.session_state.setdefault("ad_tagging_usage", {})
        with st.spinner("LLM-Tagging (Ads) läuft…"):
            subset = df[["id","ad_text","platforms","start_date","end_date","regions_count"]].copy()
            tag_df = tag_ads(subset, api_key=api_key_ui, model=model_ui)  # nutzt scripts/ad_tagger.py

            # Listen/Float sicherstellen
            tag_df["topics"] = tag_df["topics"].apply(lambda v: v if isinstance(v, list) else ( [v] if pd.notna(v) else [] ))
            tag_df["strategy"] = tag_df["strategy"].apply(lambda v: v if isinstance(v, list) else ( [v] if pd.notna(v) else [] ))
            if "confidence" in tag_df.columns:
                tag_df["confidence"] = pd.to_numeric(tag_df["confidence"], errors="coerce")

            # 🔧 rationale_bullets robust absichern
            if "rationale_bullets" in tag_df.columns:
                def _as_bullets(x):
                    if isinstance(x, list): return [str(b).strip() for b in x if str(b).strip()]
                    if pd.isna(x) or x is None: return []
                    return [str(x).strip()]
                tag_df["rationale_bullets"] = tag_df["rationale_bullets"].apply(_as_bullets)
            else:
                tag_df["rationale_bullets"] = [[] for _ in range(len(tag_df))]

        if tag_df.empty:
            st.info("Keine LLM-Ergebnisse zurückgegeben.")
        else:
            # Ergebnisse speichern
            os.makedirs("outputs", exist_ok=True)
            tag_df.to_json("outputs/ad_labels.json", orient="records", force_ascii=False, indent=2)

           # ---- Merge & Überschreiben (robust) ----
            # IDs vereinheitlichen
            tag_df["id"] = tag_df["id"].astype(str)
            df["id"] = df["id"].astype(str)

            # LLM-Spalten -> *_ext umbenennen
            tmp = tag_df.rename(columns={
                "topics": "topics_ext",
                "strategy": "strategy_ext",
                "confidence": "confidence_ext",
            })

            # Nur vorhandene Spalten mergen (verhindert KeyError)
            need_cols = [c for c in ["id","topics_ext","strategy_ext","confidence_ext"] if c in tmp.columns]
            df = df.merge(tmp[need_cols], on="id", how="left")

            # Überschreiben: wenn Basis fehlt -> direkt setzen; sonst nur dort ersetzen, wo *_ext Werte hat
            for base in ["topics", "strategy", "confidence"]:
                ext = f"{base}_ext"
                if ext in df.columns:
                    if base in df.columns:
                        df[base] = df[base].where(df[ext].isna(), df[ext])
                    else:
                        df[base] = df[ext]
                    df.drop(columns=[ext], inplace=True)

            # Ergebnisse in der Session halten und Render-Flag setzen
            st.session_state["ad_tagging_results"] = tag_df.copy()
            st.session_state["show_ad_table"] = True


# ---- Kampagnen: Ergebnis NUR speichern, Anzeige unten zentral ----
if run_camps:
    if not _campaigns_available:
        st.error("Die Kampagnenfunktionen sind nicht verfügbar. Prüfe scripts/campaign_classifier.py.")
    elif not api_key_ui:
        st.error("Bitte OpenAI API Key angeben.")
    else:
        st.session_state["last_model_used"] = model_ui
        # Wenn du dort einen eigenen System-Prompt nutzt, speichere ihn hier:
        st.session_state["campaign_system_prompt"] = st.session_state.get("campaign_system_prompt")  # ggf. später füllen
        st.session_state.setdefault("campaign_raw", {})
        st.session_state.setdefault("campaign_usage", {})
        with st.spinner("Kampagnen erkennen & klassifizieren (LLM)…"):
            if "campaign_auto_id" not in df.columns or df["campaign_auto_id"].isna().all():
                df = detect_campaigns(df.copy(), window_days=30, min_cluster_size=3, dist_thr=0.25)

            # WICHTIG: creative-Spalte absichern, bevor classify_campaigns aufgerufen wird
            if "creative" not in df.columns and "creative_features" in df.columns:
                df["creative"] = df["creative_features"]
            if "creative" not in df.columns:
                df["creative"] = [{} for _ in range(len(df))]

            camps_df = classify_campaigns(df.copy(), api_key=api_key_ui, model=model_ui, analyst_brief="Creative-Features mitverwenden.")

        if camps_df is None or camps_df.empty:
            st.info("Keine Kampagnen-Ergebnisse erhalten.")
        else:
            os.makedirs("outputs", exist_ok=True)
            camps_df.to_json("outputs/campaign_labels.json", orient="records", force_ascii=False, indent=2)

            # Optional: Mapping zurück auf df, falls ad_ids vorhanden
            if "ad_ids" in camps_df.columns:
                map_df = camps_df.explode("ad_ids").rename(columns={"ad_ids":"id"})
                if "campaign_id" in map_df.columns and "campaign_auto_id" not in map_df.columns:
                    map_df = map_df.rename(columns={"campaign_id": "campaign_auto_id"})
                map_df["id"] = map_df["id"].astype(str)
                df["id"]      = df["id"].astype(str)
                carry = ["id","campaign_auto_id","name_guess","primary_topics","strategy_mix","targeting_style","objective_guess","confidence"]
                carry = [c for c in carry if c in map_df.columns]
                df = df.merge(map_df[carry], on="id", how="left")

            st.session_state["campaign_results"] = camps_df.copy()
            st.session_state["show_camp_table"] = True
            st.success(f"Kampagnen-Klassifizierung abgeschlossen ({len(camps_df)} Kampagnen). ""(outputs/campaign_labels.json gespeichert)")

# ---- Schwachstellen-Analyse (LLM) ----
if run_weak:
    if not api_key_ui:
        st.error("Bitte OpenAI API Key angeben.")
    else:
        with st.spinner("Schwachstellen-Analyse läuft…"):
            subset = df[["id","ad_text","platforms","start_date","end_date","creative_features"]].copy()
            weak_df = analyze_weaknesses(subset, api_key=api_key_ui, model=model_ui)

        if weak_df is None or weak_df.empty:
            st.info("Keine Schwachstellen-Ergebnisse erhalten.")
        else:
            os.makedirs("outputs", exist_ok=True)
            weak_df.to_json("outputs/ad_weaknesses.json", orient="records", force_ascii=False, indent=2)
            st.session_state["weaknesses_results"] = weak_df.copy()
            st.success(f"Schwachstellen-Analyse abgeschlossen ({len(weak_df)} Ads).")

# Heuristik nur als Fallback
df["topics_heur"]   = df["ad_text"].apply(heuristic_topics)
df["strategy_heur"] = df.apply(heuristic_strategy, axis=1)

for col in ("topics", "strategy"):
    if col not in df.columns:
        df[col] = None

df["topics"]   = df["topics"].combine_first(df["topics_heur"])
df["strategy"] = df["strategy"].combine_first(df["strategy_heur"])


# Hauptspalten sicherstellen
for col in ("topics", "strategy"):
    if col not in df.columns:
        df[col] = None

# --- Ad-Tagging (LLM) Tabelle ---
if st.session_state.get("show_ad_table") and "ad_tagging_results" in st.session_state:
    ad_df_view = st.session_state["ad_tagging_results"].copy()
    ad_df_view["Ad-ID"] = ad_df_view["id"].astype(str)

    # rationale_bullets absichern + Kurzvorschau
    if "rationale_bullets" not in ad_df_view.columns:
        ad_df_view["rationale_bullets"] = [[] for _ in range(len(ad_df_view))]

    def _preview_bullets(x):
        if isinstance(x, list) and x:
            return " • ".join(x[:2])
        return ""

    ad_df_view["Begründung (kurz)"] = ad_df_view["rationale_bullets"].apply(_preview_bullets)

    st.subheader("Ergebnisse – Ad-Tagging (LLM)")

    if SUPPORTS_BUTTONCOL:
        # mit ButtonColumn (neuere Streamlit-Versionen)
        ad_df_view["Details"] = "🔍"
        edited = st.data_editor(
            ad_df_view[["Ad-ID","topics","strategy","confidence","Details"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Details": st.column_config.ButtonColumn("Details", help="Details anzeigen")
            },
            disabled=["Ad-ID","topics","strategy","confidence"]
        )
        # Klicks auswerten
        if isinstance(edited, dict) and "Details" in edited.get("column_clicked", {}):
            # letzte geklickte Zeile bestimmen
            row_idx = edited["column_clicked"]["Details"]
            ad_id_clicked = ad_df_view.iloc[row_idx]["Ad-ID"]
            _show_ad_details_overlay(ad_id_clicked)
    else:
        # Fallback ohne ButtonColumn: Tabelle + Auswahlfeld
        df_disp = ad_df_view.rename(columns={
            "topics": "Themen",
            "strategy": "Strategie",
            "confidence": "Konfidenz",
        })
        st.dataframe(
            df_disp[["Ad-ID", "Themen", "Strategie", "Konfidenz", "Begründung (kurz)"]],
            use_container_width=True
        )
        sel_ad = st.selectbox(
            "🔎 Details zu Ad anzeigen",
            ad_df_view["Ad-ID"].tolist(),
            key="sel_ad_for_details"
        )
        if st.button("Details öffnen", key=f"open_ad_details_{sel_ad}"):
            _show_ad_details_overlay(sel_ad)

# --- Kampagnen-Klassifizierung (LLM) Tabelle ---
if st.session_state.get("show_camp_table") and "campaign_results" in st.session_state:
    camp_df_view = st.session_state["campaign_results"].copy()
    if isinstance(camp_df_view, pd.DataFrame) and "campaign_auto_id" not in camp_df_view.columns and "campaign_id" in camp_df_view.columns:
        camp_df_view = camp_df_view.rename(columns={"campaign_id":"campaign_auto_id"})

    # Anzeige-DF vorbereiten
    show_cols = []
    if "campaign_auto_id" in camp_df_view.columns: show_cols.append("campaign_auto_id")
    if "name_guess" in camp_df_view.columns: show_cols.append("name_guess")
    if "primary_topics" in camp_df_view.columns: show_cols.append("primary_topics")
    if "strategy_mix" in camp_df_view.columns: show_cols.append("strategy_mix")
    if "confidence" in camp_df_view.columns: show_cols.append("confidence")

    # Schöne Labels
    camp_view_pretty = camp_df_view[show_cols].rename(columns={
        "campaign_auto_id": "Kampagnen-ID",
        "name_guess": "Kampagnenname (geschätzt)",
        "primary_topics": "Hauptthemen",
        "strategy_mix": "Strategiemix",
        "confidence": "Konfidenz",
    })

    st.subheader("Ergebnisse – Kampagnen-Klassifizierung (LLM)")

    if SUPPORTS_BUTTONCOL:
        # Variante mit ButtonColumn (wenn vorhanden)
        camp_view_pretty = camp_view_pretty.copy()
        camp_view_pretty["Details"] = "🔍"

        edited = st.data_editor(
            camp_view_pretty,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Details": st.column_config.ButtonColumn("Details", help="Details anzeigen")
            },
            disabled=[c for c in camp_view_pretty.columns if c != "Details"]
        )

        # Klick erkennen (je nach Streamlit-Version kann es kein Signal geben; dann unten Fallback verwenden)
        clicked_row = edited.get("column_clicked", {}).get("Details") if isinstance(edited, dict) else None
        if clicked_row is not None:
            # Kampagnen-ID der geklickten Zeile ermitteln
            cid = camp_view_pretty.iloc[int(clicked_row)]["Kampagnen-ID"]
            _show_campaign_details_overlay(str(cid))

    else:
        # Fallback ohne ButtonColumn: Tabelle + Auswahlfeld
        st.dataframe(camp_view_pretty, use_container_width=True)

        sel_cid = None
        if "Kampagnen-ID" in camp_view_pretty.columns:
            options = camp_view_pretty["Kampagnen-ID"].astype(str).tolist()
            sel_cid = st.selectbox(
                "🔎 Details zu Kampagne anzeigen",
                options,
                key="sel_campaign_for_details"   # << eindeutiger Key
            ) if options else None
        else:
            st.caption("Hinweis: Kampagnen-ID nicht vorhanden – Detailauswahl nicht möglich.")

        if sel_cid and st.button("Details öffnen", key=f"open_campaign_details_{sel_cid}"):
            _show_campaign_details_overlay(str(sel_cid))

# ---------------------------
# Anzeige – Schwachstellen (NEU)
# ---------------------------
weak_view = st.session_state.get("weaknesses_results")

if isinstance(weak_view, pd.DataFrame) and not weak_view.empty:
    # --- deutsche Labels + vorhandene Spalten robust mappen ---
    rename_map = {
        "id": "Ad-ID",
        "overall_risk": "Gesamtrisiko",
        "overall_confidence": "Gesamt-Konfidenz",

        "score_factual_accuracy":        "Faktischer Gehalt – Score",
        "score_framing_quality":         "Framing – Score",
        "score_visual_mislead":          "Visuals – Score",
        "score_targeting_risks":         "Targeting – Score",
        "score_policy_legal":            "Policy/Recht – Score",
        "score_transparency_context":    "Transparenz – Score",
        "score_consistency_history":     "Konsistenz – Score",
        "score_usability_accessibility": "Usability – Score",
    }
    rename_map_existing = {k: v for k, v in rename_map.items() if k in weak_view.columns}
    pretty = weak_view.rename(columns=rename_map_existing)

    # alle potenziellen Score-Spalten (deutsche Labels)
    score_cols_de_all = [
        "Faktischer Gehalt – Score",
        "Framing – Score",
        "Visuals – Score",
        "Targeting – Score",
        "Policy/Recht – Score",
        "Transparenz – Score",
        "Konsistenz – Score",
        "Usability – Score",
    ]
    score_cols_present = [c for c in score_cols_de_all if c in pretty.columns]

    # --- Filter-Controls (oberhalb der Tabellen) ---
    st.subheader("Ergebnisse – Schwachstellen (LLM)")

    # --- Werte aus Sidebar lesen ---
    score_threshold = float(st.session_state.get("weak_score_min", 0.50))
    conf_threshold  = float(st.session_state.get("weak_conf_min", 0.50))
    filter_mode     = st.session_state.get("weak_filter_mode", "Kategorie-Score")

    # --- Initiale Filterlogik: Ads nur zeigen, wenn irgendein Score >= threshold ---
    df_f = pretty.copy()

    def _row_has_any_score_ge_threshold(row: pd.Series, cols: list[str], thr: float) -> bool:
        vals = [row[c] for c in cols if c in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return any(float(v) >= thr for v in vals)

    if filter_mode == "Gesamtrisiko (overall_risk)":
        # nach overall_risk filtern
        mask_score_any = df_f["Gesamtrisiko"].fillna(0) >= score_threshold
    else:
        # mind. eine Kategorie ≥ Schwellenwert
        mask_score_any = (
            df_f.apply(lambda r: _row_has_any_score_ge_threshold(r, score_cols_present, score_threshold), axis=1)
            if score_cols_present else False
        )

    # optional nach Gesamt-Konfidenz
    if conf_threshold is not None and ("Gesamt-Konfidenz" in df_f.columns):
        mask_conf = df_f["Gesamt-Konfidenz"].fillna(0) >= conf_threshold
    else:
        mask_conf = True

    df_summary = df_f[mask_score_any & mask_conf].copy()

    # Fallback: wenn keine Score-Spalten vorhanden, wenigstens Gesamtrisiko anzeigen
    cols_summary = ["Ad-ID", "Gesamtrisiko"]
    if "Gesamt-Konfidenz" in df_summary.columns:
        cols_summary.append("Gesamt-Konfidenz")
    cols_summary += score_cols_present

    # Infozeile
    st.caption(
        f"Zeige {len(df_summary)} von {len(df_f)} Ads "
        f"(Score ≥ {score_threshold:.2f}"
        + (f", Gesamt-Konfidenz ≥ {conf_threshold:.2f}" if conf_threshold is not None and 'Gesamt-Konfidenz' in df_f.columns else "")
        + ")."
    )

    st.dataframe(df_summary[cols_summary], use_container_width=True)

    # ---------------- Details je Kategorie ----------------
    with st.expander("Details je Kategorie (Begründungen & Beispiele)", expanded=True):
        # Prüfen, ob es per-Kategorie-Konfidenzen gibt
        has_cat_conf = any(col.startswith("confidence_") for col in weak_view.columns)

        # Kurzhilfe für Beispiele in einer Zelle
        def _short(x):
            if isinstance(x, list):
                return "; ".join([str(it)[:120] for it in x])[:500]
            return x

        cat_map_de = {
            "factual_accuracy": "Faktischer Gehalt",
            "framing_quality": "Framing",
            "visual_mislead": "Visuals",
            "targeting_risks": "Targeting",
            "policy_legal": "Policy/Recht",
            "transparency_context": "Transparenz",
            "consistency_history": "Konsistenz",
            "usability_accessibility": "Usability",
        }

        # Long-Format aus *weak_view* (nicht aus pretty), damit wir auf confidence_<cat> zugreifen können
        long_rows = []
        for _, r in weak_view.iterrows():
            for k, k_de in cat_map_de.items():
                long_rows.append({
                    "Ad-ID": r.get("id"),
                    "Kategorie": k_de,
                    "Score": r.get(f"score_{k}"),
                    "Konfidenz": r.get(f"confidence_{k}") if has_cat_conf else None,
                    "Begründung": r.get(f"rationale_{k}"),
                    "Beispiele": _short(r.get(f"examples_{k}")),
                    "overall_conf": r.get("overall_confidence"),
                })
        long_df = pd.DataFrame(long_rows)

       # Filtern: je Kategorie-Score; zusätzlich optional Gesamt-Konfidenz der Ad
        mask_score = long_df["Score"].fillna(0) >= score_threshold
        if conf_threshold is not None and "overall_conf" in long_df.columns:
            mask_conf2 = long_df["overall_conf"].fillna(0) >= conf_threshold
        else:
            mask_conf2 = True

        long_filtered = long_df[mask_score & mask_conf2].copy()

        st.caption(
            f"Zeige {len(long_filtered)} von {len(long_df)} Zeilen "
            f"(Score ≥ {score_threshold:.2f}"
            + (f", Gesamt-Konfidenz ≥ {conf_threshold:.2f}" if conf_threshold is not None else "")
            + ")."
        )

        # Anzeige
        st.dataframe(long_filtered.drop(columns=["overall_conf"], errors="ignore"), use_container_width=True)

        # Chart: Ad-IDs lesbar
        try:
            plot_df = long_filtered.copy()
            plot_df["Ad-ID"] = plot_df["Ad-ID"].astype(str)      # Strings statt Zahlen
            chart_src = plot_df.pivot_table(
                index="Ad-ID", columns="Kategorie", values="Score", aggfunc="mean", fill_value=0
            ).reset_index().melt(id_vars="Ad-ID", var_name="Kategorie (Score)", value_name="Score")
            chart_src["Ad-ID"] = chart_src["Ad-ID"].astype(str)
            fig = px.bar(chart_src, x="Ad-ID", y="Score", color="Kategorie (Score)",
                        title="Risiko-Profile pro Ad (gefiltert)")
            fig.update_layout(height=360, margin=dict(l=10, r=10, b=10, t=40))
            fig.update_xaxes(type="category")  # verhindert 1.9e15 usw.
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
# -----------------------------
# Filter
# -----------------------------
st.sidebar.header("Filter")

with st.sidebar.expander("Schwachstellen-Filter", expanded=True):
    # Defaults einmalig setzen
    st.session_state.setdefault("weak_filter_mode", "Kategorie-Score")
    st.session_state.setdefault("weak_conf_min", 0.50)
    st.session_state.setdefault("weak_score_min", 0.50)

    st.radio(
        "Filter bezieht sich auf …",
        options=["Kategorie-Score", "Gesamtrisiko (overall_risk)"],
        index=0, key="weak_filter_mode",
        help="Kategorie-Score: mind. eine Kategorie ≥ Schwelle. "
             "Gesamtrisiko: overall_risk der Ad ≥ Schwelle."
    )
    st.slider("Mindest-Konfidenz (optional)", 0.0, 1.0, key="weak_conf_min", step=0.05)
    st.slider("Mindest-Score (Schwäche)", 0.0, 1.0, key="weak_score_min", step=0.05)
    st.caption("Nur anzeigen, wenn Score ≥ Mindest-Score und "
               "Konfidenz ≥ Mindest-Konfidenz (falls vorhanden).")

# -----------------------------
# Alert-Regel-Editor (Sidebar)
# -----------------------------
st.sidebar.header("Alert-Regeln")

with st.sidebar.expander("Strategie-/Themen-Alerts", expanded=False):
    N_days_shift = st.number_input("Fenstergröße (Tage) für Themen-Shift", min_value=3, max_value=14, value=5, step=1)
    kl_threshold = st.number_input("KL-Schwelle für Themen-Shift", min_value=0.01, max_value=1.0, value=0.05, step=0.01, format="%.2f")
    # „Neues Topic >X % binnen N Tagen“ (emergence)
    emerg_threshold_pp = st.number_input("Neues Topic: Anteil zuletzt ≥ (in %)", min_value=1.0, max_value=50.0, value=10.0, step=0.5, format="%.1f")
    emerg_prev_max_pp  = st.number_input("Neues Topic: Anteil vorher ≤ (in %)", min_value=0.0, max_value=20.0, value=1.0, step=0.5, format="%.1f")

with st.sidebar.expander("Kommunikationsdruck-Alerts", expanded=False):
    last_window = st.number_input("Letztes Fenster (Tage)", min_value=3, max_value=14, value=4, step=1)
    prev_window = st.number_input("Vergleichsfenster davor (Tage)", min_value=3, max_value=14, value=4, step=1)
    spike_ratio = st.number_input("Spike-Schwelle (x-fach)", min_value=1.1, max_value=5.0, value=1.4, step=0.1, format="%.1f")


# 1) Start/End solide in echte date-Werte gießen
df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce").dt.date

# Optional: Zeilen ohne Datum entfernen (oder unten per Fallback behandeln)

from datetime import date as _date
def _safe_min(s: pd.Series):
    vals = [v for v in s.dropna().tolist() if isinstance(v, _date)]
    return min(vals) if vals else _date.today()

def _safe_max(s: pd.Series):
    vals = [v for v in s.dropna().tolist() if isinstance(v, _date)]
    return max(vals) if vals else _date.today()

min_d = _safe_min(df["start_date"])
max_d = _safe_max(df["end_date"])
if max_d < min_d:
    max_d = min_d  # Reihenfolge absichern

actors = sorted(df["advertiser"].dropna().unique().tolist())
sel_actors = st.sidebar.multiselect("Akteur(e)", options=actors, default=actors)

plats_all = sorted(set(sum([x.split(", ") for x in df["platforms"].fillna("").tolist()], [])))
sel_plats = st.sidebar.multiselect("Plattform(en)", options=plats_all, default=plats_all or None)

# 2) Date-Range robust lesen (Widget kann auch single-date zurückgeben)
dr_val = st.sidebar.date_input("Zeitraum", value=(min_d, max_d))
if isinstance(dr_val, tuple) and len(dr_val) == 2:
    dr_start, dr_end = dr_val
else:
    dr_start = dr_val
    dr_end = dr_val

# Fallbacks, falls None/NaT reinrutscht
dr_start = dr_start or min_d
dr_end   = dr_end or max_d
if dr_end < dr_start:
    dr_end = dr_start

def plat_match(ps, selected):
    s = set((ps or "").split(", "))
    return bool(s & set(selected)) if selected else True

mask = (
    df["advertiser"].isin(sel_actors)
    & df["start_date"].apply(lambda x: (x or dr_start) <= dr_end)
    & df["end_date"].apply(lambda x: (x or dr_end)   >= dr_start)
    & df["platforms"].apply(lambda x: plat_match(x, sel_plats))
)

dff = df[mask].copy()


# -----------------------------
# Alerts (Shifts & Spikes)
# -----------------------------
st.subheader("Früherkennung & Alerts")

daily = (
    pd.DataFrame()
    if dff.empty
    else pd.concat(
        [
            pd.DataFrame(
                {
                    "date": pd.date_range(r["start_date"], r["end_date"], freq="D"),
                    "id": r["id"],
                    "campaign": r["campaign"],
                    "spend": (r["spend_mid"] or 0) / max(1, (r["end_date"] - r["start_date"]).days + 1),
                    "impressions": (r["impressions_mid"] or 0) / max(1, (r["end_date"] - r["start_date"]).days + 1),
                    "topics": [r["topics"]] * max(1, (r["end_date"] - r["start_date"]).days + 1),
                }
            )
            for _, r in dff.iterrows()
        ],
        ignore_index=True,
    )
)

alerts = []
topics_ts = pd.DataFrame()

if not daily.empty:
    # Topic distribution (Spend-Anteile je Tag & Topic)
    rows_ts = []
    for _, r in dff.iterrows():
        days = max(1, (r["end_date"] - r["start_date"]).days + 1)
        per_day_spend = (r["spend_mid"] or 0) / days
        topics = r["topics"] or ["Sonstiges"]
        for i in range(days):
            dt = r["start_date"] + timedelta(days=i)
            for t in topics:
                rows_ts.append({"date": dt, "topic": t, "spend": per_day_spend / len(topics)})
    topics_ts = pd.DataFrame(rows_ts)

    # --- Strategischer Shift (KL, N=N_days_shift) + Details & Emerging ---
    N = int(N_days_shift)
    if not topics_ts.empty and topics_ts["date"].nunique() >= 2 * N:
        agg = topics_ts.groupby(["date", "topic"], as_index=False)["spend"].sum()
        dates_sorted = sorted(agg["date"].unique())
        last_dates, prev_dates = dates_sorted[-N:], dates_sorted[-2 * N : -N]

        p = agg[agg["date"].isin(last_dates)].groupby("topic")["spend"].sum()
        q = agg[agg["date"].isin(prev_dates)].groupby("topic")["spend"].sum()
        keys = sorted(set(p.index).union(q.index))
        P = np.array([p.get(k, 0.0) for k in keys])
        Q = np.array([q.get(k, 0.0) for k in keys])

        if P.sum() > 0 and Q.sum() > 0:
            P = P / P.sum()
            Q = Q / Q.sum()
            kld = kl_div(P, Q)
            delta = pd.Series(P - Q, index=keys).sort_values(key=np.abs, ascending=False)

            if kld > float(kl_threshold):
                msg = ", ".join([f"{k} ({'+' if v>0 else ''}{v*100:.1f}pp)" for k, v in delta.head(3).items()])
                alerts.append(("🧭 Strategischer Shift", f"Themenmix veränderte sich: {msg} (KL={kld:.3f})"))

                # Details anzeigen
                st.markdown("**Themen-Shift – Details (letzte {} Tage vs. davor)**".format(N))
                details_df = (
                    pd.DataFrame(
                        {
                            "Topic": keys,
                            "Anteil vorher (%)": Q * 100,
                            "Anteil zuletzt (%)": P * 100,
                            "Δ Prozentpunkte": (P - Q) * 100,
                        }
                    )
                    .sort_values("Δ Prozentpunkte", key=lambda s: s.abs(), ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(details_df, use_container_width=True)

                # Balkenplot Δpp
                fig_delta = px.bar(
                    details_df.head(10),
                    x="Topic",
                    y="Δ Prozentpunkte",
                    title="Top-10 Themen-Verschiebungen (Δ Prozentpunkte)",
                )
                fig_delta.update_layout(height=320, margin=dict(l=10, r=10, b=10, t=40))
                st.plotly_chart(fig_delta, use_container_width=True)

                # Emerging Topic: neu ≥ emerg_threshold_pp & vorher ≤ emerg_prev_max_pp
                emer_mask = (P * 100 >= emerg_threshold_pp) & (Q * 100 <= emerg_prev_max_pp)
                emerging = [keys[i] for i, ok in enumerate(emer_mask) if ok]
                if emerging:
                    alerts.append((
                        "🆕 Neues Topic",
                        f"Neu aufgetaucht/hochgesprungen: {', '.join(emerging)} (≥{emerg_threshold_pp:.1f}% in den letzten {N} Tagen)"
                    ))

    # --- Kommunikationsdruck-Spike (konfigurierbar) + Treiber-Ads ---
    spend_by_day = daily.groupby("date", as_index=False)["spend"].sum().sort_values("date")
    if len(spend_by_day) >= (int(last_window) + int(prev_window)):
        last_mean = spend_by_day.tail(int(last_window))["spend"].mean()
        prev_mean = spend_by_day.tail(int(last_window) + int(prev_window)).head(int(prev_window))["spend"].mean()
        ratio = (last_mean / prev_mean) if prev_mean > 0 else np.inf

        if prev_mean > 0 and ratio >= float(spike_ratio):
            alerts.append((
                "📈 Kommunikationsdruck-Spike",
                f"Ø Spend letzte {int(last_window)} Tage {last_mean:,.0f} vs. davor {prev_mean:,.0f} (x{ratio:.2f})".replace(",", "."),
            ))

            # Treiber identifizieren für diese Fenster
            last_days = spend_by_day["date"].tail(int(last_window)).tolist()
            prev_days = spend_by_day["date"].tail(int(last_window) + int(prev_window)).head(int(prev_window)).tolist()

            per_ad_day = daily.groupby(["id", "campaign", "date"], as_index=False)["spend"].sum()
            by_ad_prev = (
                per_ad_day[per_ad_day["date"].isin(prev_days)]
                .groupby(["id", "campaign"], as_index=False)["spend"].sum()
                .rename(columns={"spend": "spend_prev"})
            )
            by_ad_last = (
                per_ad_day[per_ad_day["date"].isin(last_days)]
                .groupby(["id", "campaign"], as_index=False)["spend"].sum()
                .rename(columns={"spend": "spend_last"})
            )

            ad_delta = pd.merge(by_ad_last, by_ad_prev, on=["id", "campaign"], how="outer").fillna(0.0)
            ad_delta["Δ Spend"] = ad_delta["spend_last"] - ad_delta["spend_prev"]
            ad_delta = ad_delta.sort_values("Δ Spend", ascending=False)

            st.markdown("**Kommunikationsdruck – Treiber-Anzeigen (Δ Spend)**")
            st.dataframe(ad_delta.head(10), use_container_width=True)

            fig_ad = px.bar(
                ad_delta.head(10),
                x="campaign",
                y="Δ Spend",
                hover_data=["id", "spend_last", "spend_prev"],
                title=f"Top-Treiber-Anzeigen (Δ Spend: letzte {int(last_window)} Tage vs. davor {int(prev_window)})",
            )
            fig_ad.update_layout(height=320, xaxis_tickangle=-20, margin=dict(l=10, r=10, b=10, t=40))
            st.plotly_chart(fig_ad, use_container_width=True)

# Alerts ausgeben
if alerts:
    for a in alerts:
        st.success(f"**{a[0]}** – {a[1]}")
else:
    st.info("Keine signifikanten Shifts/Spikes erkannt (Heuristik).")

# -----------------------------
# Qualitativ
# -----------------------------
st.subheader("Qualitatives Monitoring – Themen & Strategie")
if not dff.empty:
    # Themenmix über Zeit
    if 'topics_ts' in locals() and not topics_ts.empty:
        area = topics_ts.groupby(["date","topic"], as_index=False)["spend"].sum()
        fig = px.area(area, x="date", y="spend", color="topic", title="Themenmix über Zeit (Spend-gewichtete Heuristik)")
        fig.update_layout(height=360, legend_orientation="h", margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Strategie (heuristisch) je Anzeige")
    for _, r in dff.iterrows():
        st.markdown(f"- **{r['campaign']}** · {r['platforms']}  \n  " + " ".join([f"`{s}`" for s in r['strategy']]))

# -----------------------------
# Quantitativ
# -----------------------------
st.subheader("Quantitatives Monitoring – Timing & Druck")
if not daily.empty:
    fig1 = px.line(daily.groupby("date", as_index=False)["spend"].sum(), x="date", y="spend", title="Ausgaben über Zeit")
    fig1.update_layout(height=320, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.area(daily.groupby("date", as_index=False)["impressions"].sum(), x="date", y="impressions", title="Impressions über Zeit")
    fig2.update_layout(height=320, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig2, use_container_width=True)

    tl = daily.groupby(["id","campaign"], as_index=False).agg(start=("date","min"), end=("date","max"), spend=("spend","sum"))
    if not tl.empty:
        fig3 = px.timeline(tl, x_start="start", x_end="end", y="campaign", color="spend", title="Kampagnen-Timeline")
        fig3.update_yaxes(autorange="reversed")
        fig3.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig3, use_container_width=True)

# Plattform-Budget
by_plat = dff.assign(platform=lambda x: x["platforms"].str.split(", ")).explode("platform").groupby("platform", as_index=False)["spend_mid"].sum()
if not by_plat.empty:
    fig4 = px.bar(by_plat, x="platform", y="spend_mid", title="Budget nach Plattform")
    fig4.update_layout(height=320, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig4, use_container_width=True)
# -----------------------------
# Zielgruppen & Regionen
# -----------------------------
st.subheader("Zielgruppen & Regionen")

# Zielgruppen (geschätzt über demographische Anteile)
aud_rows = []
for _, r in dff.iterrows():
    dem = r["gender_age_json"] if isinstance(r["gender_age_json"], list) else []
    s = r["spend_mid"] or 0.0
    for dmp in dem:
        aud_rows.append({
            "audience": f"{dmp.get('age','?')} · {dmp.get('gender','?')}",
            "spend": s * (float(dmp.get('percentage')) if dmp.get('percentage') is not None else np.nan),
        })
aud_df = pd.DataFrame(aud_rows)
if not aud_df.empty:
    fig5 = px.bar(aud_df.groupby("audience", as_index=False)["spend"].sum().sort_values("spend", ascending=False),
                  x="audience", y="spend", title="Budget nach Zielgruppe (heuristisch)")
    fig5.update_layout(height=360, xaxis_tickangle=-25, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig5, use_container_width=True)

# Regionen
reg_rows = []
for _, r in dff.iterrows():
    regs = r["regions_json"] if isinstance(r["regions_json"], list) else []
    s = r["spend_mid"] or 0.0
    for rg in regs:
        reg_rows.append({"region": rg.get("region","?"), "spend": s * (float(rg.get('percentage')) if rg.get('percentage') is not None else np.nan)})
reg_df = pd.DataFrame(reg_rows)
if not reg_df.empty:
    fig6 = px.bar(reg_df.groupby("region", as_index=False)["spend"].sum().sort_values("spend", ascending=False),
                  x="region", y="spend", title="Regionale Schwerpunkte (heuristisch)")
    fig6.update_layout(height=360, xaxis_tickangle=-25, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig6, use_container_width=True)

# -----------------------------
# Regionen – Heatmap (Bubble-Heat über BL-Zentroiden)
# -----------------------------
bl_centroids = {
    "Baden-Württemberg": (48.6616, 9.3501),
    "Bayern": (48.7904, 11.4979),
    "Berlin": (52.5200, 13.4050),
    "Brandenburg": (52.4125, 12.5316),
    "Bremen": (53.0793, 8.8017),
    "Hamburg": (53.5511, 9.9937),
    "Hessen": (50.6521, 9.1624),
    "Mecklenburg-Vorpommern": (53.6127, 12.4296),
    "Niedersachsen": (52.6367, 9.8451),
    "Nordrhein-Westfalen": (51.4332, 7.6616),
    "Rheinland-Pfalz": (49.9929, 7.8460),
    "Saarland": (49.3964, 6.9770),
    "Sachsen": (51.1045, 13.2017),
    "Sachsen-Anhalt": (51.9503, 11.6923),
    "Schleswig-Holstein": (54.2194, 9.6961),
    "Thüringen": (50.9010, 11.0375),
}
rename_map = {"Saxony-Anhalt": "Sachsen-Anhalt"}  # evtl. engl. Namen normalisieren

# Quelle wählen: bevorzugt reg_df (bereits spend-gewichtet), sonst Fallback aus df["regions_json"]
if 'reg_df' in locals() and not reg_df.empty:
    reg_sum = reg_df.groupby("region", as_index=False)["spend"].sum()
else:
    # Fallback: aus df["regions_json"] + spend_mid konstruieren (ohne LLM!)
    reg_rows = []
    if "regions_json" in df.columns and not df["regions_json"].dropna().empty:
        for _, r in df.iterrows():
            regs = r.get("regions_json") or []
            s = float(r.get("spend_mid") or 0.0)
            for rg in regs:
                try:
                    p = float(rg.get("percentage")) if rg.get("percentage") is not None else np.nan
                except Exception:
                    p = np.nan
                reg_rows.append({
                    "region": rg.get("region", "?"),
                    "spend": s * p if pd.notna(p) else np.nan
                })

    reg_sum = (
        pd.DataFrame(reg_rows)
        .groupby("region", as_index=False)["spend"].sum()
    ) if reg_rows else pd.DataFrame(columns=["region", "spend"])

# Namen bereinigen
if not reg_sum.empty:
    reg_sum["region_clean"] = reg_sum["region"].replace(rename_map)

# Map vorbereiten
rows_map = []
for _, row in reg_sum.iterrows():
    rname = row.get("region_clean")
    if rname in bl_centroids:
        lat, lon = bl_centroids[rname]
        rows_map.append({"region": rname, "lat": lat, "lon": lon, "spend": row["spend"]})

if rows_map:
    map_df = pd.DataFrame(rows_map)
    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lon",
        size="spend",
        hover_name="region",
        hover_data={"spend":":.0f", "lat":False, "lon":False},
        zoom=5, height=520,
        title="Regionale Heatmap (Bubble-Intensity nach Spend)"
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.caption("Keine passenden Regionen für Heatmap gefunden.")

# ---- Label-Helfer NUR EINMAL: Für Ad-Ansichten -> \"Ad <id>\", für Kampagnen-Ansichten -> LLM-Name bevorzugt ----
def _row_label_for_ad(r: pd.Series) -> str:
    rid = r.get("id")
    return f"Ad {rid}" if pd.notna(rid) else "Ad –"

def _row_label_for_campaign(r: pd.Series) -> str:
    # bevorzugt LLM-Name, sonst vorhandener Kampagnentitel, sonst ID
    if isinstance(r.get("campaign_name_llm"), str) and r["campaign_name_llm"].strip():
        return r["campaign_name_llm"].strip()
    if isinstance(r.get("campaign"), str) and r["campaign"].strip():
        return r["campaign"].strip()
    if isinstance(r.get("campaign_auto_id"), str):
        return r["campaign_auto_id"]
    if isinstance(r.get("campaign_id"), str):
        return r["campaign_id"]
    return "Kampagne –"

# -----------------------------
# Kreativ-Analyse (optional)
# -----------------------------
st.subheader("Kreativ-Analyse (optional aus Screenshot-Features)")

# --- Ebene wählen: Ad oder Kampagne (falls LLM-Kampagnen da sind) ---
has_camp = isinstance(st.session_state.get("campaign_results"), pd.DataFrame) \
           and not st.session_state["campaign_results"].empty

ana_mode = st.radio(
    "Analyse-Ebene",
    options=["Ad-Ebene", "Kampagnen-Ebene (LLM)"] if has_camp else ["Ad-Ebene"],
    horizontal=True,
)

# --- Mapping Ad-ID -> Kampagnen-ID/Name aus LLM-Ergebnis herstellen ---
ad2camp = {}
camp_name_by_id = {}
if has_camp:
    camps_df = st.session_state["campaign_results"].copy()
    if "campaign_auto_id" not in camps_df.columns and "campaign_id" in camps_df.columns:
        camps_df = camps_df.rename(columns={"campaign_id": "campaign_auto_id"})
    for _, row in camps_df.iterrows():
        cid = str(row.get("campaign_auto_id") or "")
        cname = row.get("name_guess")
        if isinstance(row.get("ad_ids"), (list, tuple)):
            for aid in row["ad_ids"]:
                ad2camp[str(aid)] = cid
        camp_name_by_id[cid] = cname if isinstance(cname, str) and cname.strip() else None

if ad2camp:
    dff["id"] = dff["id"].astype(str)
    dff["campaign_auto_id"] = dff["id"].map(ad2camp)
    dff["campaign_name_llm"] = dff["campaign_auto_id"].map(camp_name_by_id)

# -------- Kreativ-Daten vorbereiten (je nach Ebene) --------
base_rows = []

if ana_mode == "Ad-Ebene":
    # eine Zeile pro Ad
    for _, r in dff.iterrows():
        feats = r.get("creative_features")
        # Fallback, falls Features in additional_data liegen
        if not feats and isinstance(getattr(r, "additional_data", None), dict):
            add = r["additional_data"]
            feats = {
                "visuelle_features": add.get("visuelle_features", {}) or {},
                "textuelle_features": add.get("textuelle_features", {}) or {},
                "semantische_features": add.get("semantische_features", {}) or {},
            }
        if isinstance(feats, dict) and any(
            isinstance(feats.get(k), dict) and feats.get(k) for k in
            ["visuelle_features", "textuelle_features", "semantische_features"]
        ):
            base_rows.append({
                "label": _row_label_for_ad(r),           # << Ad-ID als Label
                "level": "ad",
                "id": r.get("id"),
                "campaign": r.get("campaign"),
                "campaign_auto_id": r.get("campaign_auto_id"),
                "campaign_name_llm": r.get("campaign_name_llm"),
                "spend_mid": r.get("spend_mid") or 0.0,
                "vis": (feats.get("visuelle_features") or {}),
                "txt": (feats.get("textuelle_features") or {}),
                "sem": (feats.get("semantische_features") or {}),
            })

else:
    # Kampagnen-Ebene: alle Ads pro Kampagne zusammenfassen
    from collections import Counter
    if "campaign_auto_id" in dff.columns:
        for cid, grp in dff.groupby("campaign_auto_id"):
            if pd.isna(cid):
                continue

            spends = []
            flaechen = []  # (text,bild,weißraum, spend)
            ctas = []
            sems = []
            pals = []

            for _, r in grp.iterrows():
                feats = r.get("creative_features")
                if not feats and isinstance(getattr(r, "additional_data", None), dict):
                    add = r["additional_data"]
                    feats = {
                        "visuelle_features": add.get("visuelle_features", {}) or {},
                        "textuelle_features": add.get("textuelle_features", {}) or {},
                        "semantische_features": add.get("semantische_features", {}) or {},
                    }
                s = float(r.get("spend_mid") or 0.0)
                spends.append(s)
                if isinstance(feats, dict):
                    vis = feats.get("visuelle_features", {}) or {}
                    txt = feats.get("textuelle_features", {}) or {}
                    sem = feats.get("semantische_features", {}) or {}

                    fl = (vis.get("flächenverteilung") or {})
                    if fl:
                        flaechen.append((fl.get("textfläche", np.nan),
                                         fl.get("bildfläche", np.nan),
                                         fl.get("weißraum", np.nan), s))
                    ctas.append((txt.get("cta_typ", "?"),
                                 txt.get("cta_visuelle_prominenz", "?"),
                                 txt.get("cta_position", "?")))
                    sems.append((sem.get("argumenttyp", "?"),
                                 sem.get("emotionaler_apell", "?"),
                                 sem.get("framing_typ", "?")))
                    pals.extend(vis.get("farbpalette", []) or [])

            # gewichtetes Mittel der Flächenverteilung
            def _wmean(vals):
                num = sum(v*s for v, s in vals if pd.notna(v))
                den = sum(s for v, s in vals if pd.notna(v))
                return num/den if den > 0 else np.nan

            text_ratio  = _wmean([(t, s) for (t, b, w, s) in flaechen])
            bild_ratio  = _wmean([(b, s) for (t, b, w, s) in flaechen])
            weiss_ratio = _wmean([(w, s) for (t, b, w, s) in flaechen])

            # Modus
            def _mode(lst, idx):
                c = Counter([x[idx] for x in lst if x and x[idx] is not None])
                return c.most_common(1)[0][0] if c else "?"

            cta_typ  = _mode(ctas, 0)
            cta_prom = _mode(ctas, 1)
            cta_pos  = _mode(ctas, 2)
            argtyp   = _mode(sems, 0)
            apell    = _mode(sems, 1)
            framing  = _mode(sems, 2)

            top_cols = [c for c, _cnt in Counter(pals).most_common(4)]

            base_rows.append({
                "label": camp_name_by_id.get(cid) or str(cid),  # << Kampagnenname
                "level": "campaign",
                "campaign_auto_id": cid,
                "campaign_name_llm": camp_name_by_id.get(cid),
                "spend_mid": sum(spends),
                "vis": {"flächenverteilung": {"textfläche": text_ratio,
                                              "bildfläche": bild_ratio,
                                              "weißraum": weiss_ratio},
                        "farbpalette": top_cols},
                "txt": {"cta_typ": cta_typ,
                        "cta_visuelle_prominenz": cta_prom,
                        "cta_position": cta_pos},
                "sem": {"argumenttyp": argtyp,
                        "emotionaler_apell": apell,
                        "framing_typ": framing},
            })

cre = pd.DataFrame(base_rows)

# -------- Visualisierung / Tabellen (mit 'label') --------
if not cre.empty:
    # Flächenverteilung
    ratios = []
    for _, r in cre.iterrows():
        fl = (r["vis"] or {}).get("flächenverteilung") or {}
        if fl:
            ratios.append({
                "Eintrag": r["label"],
                "Text": fl.get("textfläche", np.nan),
                "Bild": fl.get("bildfläche", np.nan),
                "Weißraum": fl.get("weißraum", np.nan),
            })
    if ratios:
        rdf = pd.DataFrame(ratios).melt(id_vars="Eintrag", var_name="Typ", value_name="Anteil")
        fig7 = px.bar(
            rdf, x="Eintrag", y="Anteil", color="Typ", barmode="stack",
            title="Flächenverteilung pro Ad" if ana_mode == "Ad-Ebene" else "Flächenverteilung pro Kampagne",
        )
        fig7.update_layout(height=360, xaxis_tickangle=-15, margin=dict(l=10, r=10, b=10, t=40))
        st.plotly_chart(fig7, use_container_width=True)

    # Farbpaletten (Preview)
    pal_rows = []
    for _, r in cre.iterrows():
        pals = (r["vis"] or {}).get("farbpalette") or []
        if pals:
            pal_rows.append({"Eintrag": r["label"], "palette": pals})

    if pal_rows:
        st.markdown("**Farbpaletten (Preview)**")
        for row in pal_rows:
            cells = "".join([
                f"<span style='display:inline-block;width:18px;height:18px;border-radius:3px;margin-right:6px;background:{c}' title='{c}'></span>"
                for c in row["palette"]
            ])
            st.markdown(f"{row['Eintrag']}: {cells}", unsafe_allow_html=True)

    # CTA-Übersicht
    cta = []
    for _, r in cre.iterrows():
        t = r["txt"] or {}
        cta.append({
            "Eintrag": r["label"],
            "CTA-Typ": t.get("cta_typ", "?"),
            "Prominenz": t.get("cta_visuelle_prominenz", "?"),
            "Position": t.get("cta_position", "?"),
        })
    st.dataframe(pd.DataFrame(cta), use_container_width=True)

    # Semantik
    sem_rows = []
    for _, r in cre.iterrows():
        s = r["sem"] or {}
        sem_rows.append({
            "Eintrag": r["label"],
            "Argumenttyp": s.get("argumenttyp", "?"),
            "Apell": s.get("emotionaler_apell", "?"),
            "Framing": s.get("framing_typ", "?"),
        })
    st.dataframe(pd.DataFrame(sem_rows), use_container_width=True)

else:
    st.caption("Keine Creative-Features übergeben – Abschnitt übersprungen.")

# -----------------------------
# LLM – Prompt Scaffold (keine Live-Calls)
# -----------------------------
# st.subheader("LLM-Integration – Prompts (zum Kopieren)")
# topic_prompt = """Klassifiziere Anzeigen-Texte in Themen (max. 3) und gib JSON zurück: [{id, topics:[...] }].
# Kategorien: ["Klima & Energie","Mobilität & ÖV","Soziales & Verteilung","Migration & Integration","Sicherheit & Ordnung","Wirtschaft & Innovation","Außen & Entwicklung","Sonstiges"].
# Input: { "id":"<AD_ID>", "text":"<AD_TEXT>", "meta": { "regions":[...], "gender_age":[...], "platforms":"...", "period":["<start>","<end>"] } }"""
# strategy_prompt = """Analysiere je Anzeige Strategie (Hypertargeting, Mobilisierung, Persuasion, Kontrastierung).
# Gib JSON zurück: [{id, strategy:[...], confidence, rationale}]. Nutze Meta-Daten (Zielgruppen, Regionen, Timing, Budget) + Text."""
# timing_prompt = """Interpretiere Ausgaben über Zeit/Plattform/Region/Zielgruppe. Gib 3 Bullet-Insights + 2 Forschungsfragen.
# Beurteile, ob Hypertargeting/Mobilisierung/Persuasion dominiert und begründe kurz."""
# st.code(topic_prompt, language="text")
# st.code(strategy_prompt, language="text")
# st.code(timing_prompt, language="text")

# -----------------------------
# Datenvorschau
# -----------------------------
st.subheader("Datenvorschau")
st.dataframe(dff[["id","advertiser","platforms","campaign","start_date","end_date","spend_mid","impressions_mid","topics","strategy","snapshot_url"]], use_container_width=True)

st.markdown("""---
**Hinweise**  
- Meta-Limits (Spannen für Spend/Impressions) werden via Midpoint visualisiert.  
- Alerts: Themen-Shift (KL-Divergenz, N=5) & Spend-Spike (x1.4) – Sensitivität anpassbar.  
- Creative-Features: Falls vorhanden, werden Layout/CTA/Semantik visualisiert.  
""")
