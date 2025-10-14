import os
import json
import datetime
from openai import OpenAI
import openai

# Initialisiere OpenAI-Client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt-Text für Bildanalyse
prompt_text = """Antworte ausschließlich mit **valider JSON-Struktur** im folgenden Format. Beginne mit `{` und schließe mit `}`. 
Gib keine Einleitung, kein Markdown, keine Formatierungen, keine Kommentare aus.
Du bist ein spezialisiertes KI-Modell zur tiefgreifenden Analyse von Online-Werbeanzeigen – auch in Form von Screenshots aus sozialen Medien (z. B. Facebook, Instagram, LinkedIn, Google Ads, TikTok).

Analysiere das folgende Werbebild oder den Screenshot **systematisch und vollständig**. Berücksichtige dabei sowohl **visuelle Merkmale**, **textliche Inhalte**, **semantische Strategien** als auch **plattformtypische Elemente** (z. B. UI-Design, Kommentarfelder, Buttons, Reaktionen).

Untersuche insbesondere:

1. **Alle sichtbaren Elemente im Bild**
   - Beschreibe: Typ, Position, Farbe, Form, Größe, Stil, Funktion, Bedeutung
   - Erkenne: CTA-Buttons, Icons, Rabatt-Sticker, Social-Media-UI, Logos, Personen, Produkte, Layout-Raster etc.
   - Gib zu jedem Element an: Was ist es? Wo ist es? Wozu dient es? Wie wirkt es?

2. **Textebene vollständig analysieren**
   - Extrahiere **alle Textblöcke** vollständig (auch UI-Texte, Kommentare, Randnotizen)
   - Gib pro Textblock an:
     - Inhalt (Wortlaut)
     - Funktion (z. B. CTA, Branding, Info, Rabatt)
     - Sprachebene (formell, werblich, neutral etc.)
     - Ton & Wirkung (z. B. motivierend, informierend, drängend)

3. **Quantitative Textmetriken berechnen**
   - Zeichenanzahl & Wortanzahl der Headline
   - Gesamtzeichenanzahl & Gesamtwortanzahl
   - Durchschnittliche Wortlänge
   - Anzahl unterschiedlicher Schriftarten
   - Verhältnis Textfläche zu Bildfläche (in Prozent)

4. **Screenshot-Erkennung & Plattformkontext**
   - Prüfe: Handelt es sich um einen Screenshot?
   - Indikatoren: UI-Elemente (Like-Zähler, Kommentare, Buttons, Menüs, Scrollbars)
   - Gib ggf. erkannte Plattform an: Facebook, Instagram, LinkedIn, TikTok, Google Ads etc.

5. **Visuelle Gestaltung & Layoutanalyse**
   - Farbkontraste und dominante Farben (Farbcodes)
   - Kompositionstyp: zentral / asymmetrisch / Raster etc.
   - Blickführung (zentriert, radial, dynamisch etc.)
   - Layoutstruktur: Social Feed, Kachel, Story, klassische Anzeige
   - Verhältnis Text / Bild / Weißraum (in %)
   - Schriftarten & -größenverteilung
   - Textausrichtung
   - Professionalitätsgrad des Designs

6. **Semantische & persuasive Strategie**
   - Emotionale oder rationale Appelle
   - Erkannte Symbole (z. B. Haken, Herz, Stern, Flamme)
   - Wirkung des Symbols (z. B. Vertrauen, Dringlichkeit)
   - Werbeversprechen-Typ: USP, ESP, generisch
   - Zielgruppenmerkmale anhand Bildsprache & Sprachstil
   - Framing-Typ: Gewinn, Verlust, Moralisch, Autorität, Vergleich
   - Ansprache-Typ: direkt / allgemein / duzend / siezend

Gib ausschließlich den folgenden JSON zurück – **ohne Markdown, ohne Erklärung, ohne zusätzliche Kommentare**.

JSON-Format:

{
  "visuelle_features": {
    "farbpalette": ["#FFAA00", "#000000", "#FFFFFF"],
    "schriftarten_erkannt": ["Arial", "Sans Serif"],
    "schriftgrößen_verteilung": { "klein": 2, "mittel": 1, "groß": 1 },
    "textausrichtung": "zentriert | linksbündig | rechtsbündig | gemischt",
    "flächenverteilung": { "textfläche": 23, "bildfläche": 60, "weißraum": 17 },
    "kompositionstyp": "Zentrumskomposition | asymmetrisch | Raster",
    "bildtyp": "Foto | Illustration | CGI | Stock | Screenshot",
    "blickführung": "zentral | dynamisch | radial",
    "salienzverteilung": 0.0 - 1.0,
    "dominante_layoutstruktur": "Einspaltig | mehrspaltig | Social-Feed | Werbekachel | klassisch",
    "plattformkontext_erkannt": true | false,
    "plattform": "Facebook | Instagram | Google | LinkedIn | TikTok | Unbekannt",
    "elemente": [
      {
        "element": "z.B. Person, Text, Button, Rabatt-Symbol, Like-Zähler",
        "position": "z.B. links unten, Zentrum, oben rechts",
        "farbe": "z.B. Blau, Rot",
        "größe": "klein | mittel | groß",
        "form": "rechteckig | kreisförmig | frei geformt",
        "interaktiv_erscheinung": true | false,
        "funktion": "CTA | Branding | Produktdarstellung | Textblock | Kommentar | Rabattinfo | Social Proof | UI-Element | Unklar",
        "bedeutung": "z.B. Vertrauen, Angebot, Aufforderung",
        "inhalt": "Nur bei Text oder UI (optional)",
        "person_mimik_erkannbar": "lächelt | neutral | ernst | nicht sichtbar",
        "bild_inhalt": "Person | Produkt | Symbol | App-Screenshot",
        "markenlogo_erkannt": true | false
      }
    ]
  },
  "textuelle_features": {
    "headline_länge": "z.B. 12",
    "headline_zeichenanzahl": 64,
    "headline_wortanzahl": 8,
    "gesamtzeichenanzahl": 182,
    "gesamtwortanzahl": 29,
    "durchschnittliche_wortlänge": 5.3,
    "anzahl_textblöcke": 3,
    "anzahl_schriftarten": 2,
    "text_bild_verhältnis": 18.5,
    "cta_typ": "Imperativ | Frage | Aussage",
    "cta_position": "oben | mitte | unten | mehrfach | nicht vorhanden",
    "cta_visuelle_prominenz": "hoch | mittel | gering",
    "cta_wirkungseinschätzung": "handlungsauffordernd | informierend | schwach",
    "sprachstil": "informativ | emotional | werbend",
    "tonalität": "freundlich | aggressiv | sachlich",
    "textgliederung_erkennbar": true | false,
    "wortartenverteilung": { "Substantive": 10, "Verben": 7, "Adjektive": 5, "Pronomen": 2 },
    "text_inhalte": [
      {
        "inhalt": "z.B. Jetzt teilnehmen!",
        "funktion": "CTA | Info | Branding | Rabatt",
        "sprachebene": "formell | neutral | umgangssprachlich",
        "wirkung": "drängt | motiviert | informiert | emotionalisiert"
      }
    ]
  },
  "semantische_features": {
    "argumenttyp": "rational | emotional | humorvoll",
    "bild_text_verhältnis": "redundant | komplementär | widersprüchlich",
    "symbolgebrauch": {
      "symbol_erkannt": true | false,
      "symbol_typ": "z.B. Herz, Stern, Blitz, Dollarzeichen, Haken",
      "symbol_bedeutung": "z.B. Liebe, Qualität, Energie, Preis, Vertrauen"
    },
    "werbeversprechen": "USP | ESP | generisch",
    "zielgruppe": "Eltern | Sportler | Kinder | Unternehmer | Senioren | Allgemein",
    "zielgruppen_indikatoren": ["Kinder im Bild", "Business-Vokabular", "Sportutensilien"],
    "emotionaler_apell": "Hoffnung | Freude | Angst | Dringlichkeit | Humor | Stolz | Unklar",
    "framing_typ": "Gewinn | Verlust | Moralisch | Expertenglaubwürdigkeit | Sozialer Vergleich",
    "ansprache_typ": "direkt | allgemein | duzend | siezend"
  }
}
Die Antwort muss exakt dieser JSON-Struktur folgen. Verwende exakt die vorgegebenen Schlüsselnamen. 
Wenn einzelne Informationen nicht erkennbar sind, verwende `"Unklar"` oder `false`, aber **verändere niemals die Struktur**.
"""

# Zeitstempel und Output-Ordner erstellen
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"Output_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# Kosten aufsummieren
def update_total_costs(new_cost):
    cost_file = "total_costs.json"
    if os.path.exists(cost_file):
        with open(cost_file, "r", encoding="utf-8") as f:
            total_data = json.load(f)
    else:
        total_data = {"total_costs": 0}
    total_data["total_costs"] += new_cost
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(total_data, f, indent=2, ensure_ascii=False)

# Analysefunktion für mehrere Bilder
def analyze_images(image_urls):
    total_tokens_used = 0
    total_cost = 0

    for i, image_url in enumerate(image_urls):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1500
        )

        # Inhalt bereinigen (Markdown-Wrapper entfernen)
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        elif content.startswith("```"):
            content = content.lstrip("```").rstrip("```").strip()

        # Versuch JSON zu laden
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Ungültiges JSON für Bild {image_url}, speichere Rohtext.")
            result = {"raw": content}

        # Speichern mit Zeitstempel im Dateinamen
        filename = f"output_{i}_{timestamp}.json"
        file_path = os.path.join(output_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        tokens = usage.total_tokens
        input_cost = input_tokens * 0.005 / 1000
        output_cost = output_tokens * 0.015 / 1000
        run_cost = input_cost + output_cost

        update_total_costs(run_cost)
        total_tokens_used += tokens
        total_cost += run_cost

        print(f"\n✅ Analyse für Bild {image_url} abgeschlossen.")
        print(f"📊 Tokens: {tokens} | 💵 Kosten: ${run_cost:.4f}")

    print(f"\n📈 Gesamt-Tokens: {total_tokens_used} | 💰 Gesamtkosten: ${total_cost:.4f}")

# Bild-URLs
image_urls = [
    # Grüne_Screenshots
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1036160865297465.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1273009827775112.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/293627929647359.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/549921350039850.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/e23b799e9768c3ec69779334926415c347f11f90/1977475649454352.png",

    # AfD_Screenshots
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1478380830183560.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1724364854862009.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1746974925909526.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1865927980938562.png",
    "https://raw.githubusercontent.com/rothmat/MarketingAnalytics/1932b066b56f2919b318223ae47d9ea6df481e4f/1901456560704428.png",

    # Grüne_Images
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1036160865297465_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1273009827775112_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_293627929647359_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_549921350039850_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1977475649454352_image_2.png",

    # AfD_Images
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1478380830183560_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1724364854862009_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1746974925909526_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1865927980938562_image_2.png",
    #"https://raw.githubusercontent.com/rothmat/MarketingAnalytics/d48b65a07a786ef668fb5aef362be1911677bf71/ad_1901456560704428_image_2.png"
]

# Starte Analyse
analyze_images(image_urls)
