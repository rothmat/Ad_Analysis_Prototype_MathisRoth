README — ad-db (PostgreSQL Stack & How-To)

Diese README erklärt das Starten des DB-Stacks, das Öffnen der DB aus dem Terminal, die Zugangsdaten,
das Schema, typische Workflows und nützliche Befehle.

──────────────────────────────────────────────────────────────────────────────
INHALT
- Voraussetzungen
- Schnellstart
- Zugangsdaten & Verbindungs-URIs
- DB aus dem Terminal öffnen (psql)
- Adminer (Web-UI)
- Umgebungsvariable für Skripte
- Schema-Überblick
- Typische Workflows (3 Sammler)
- Nützliche Prüf-Queries
- Backups & Restore
- Schema-Updates (Migrationen)
- Troubleshooting
──────────────────────────────────────────────────────────────────────────────

VORAUSSETZUNGEN
- Docker Desktop installiert und läuft.
- Projektstruktur (vereinfacht):
  <ProjektRoot>/
    ad-db/
      docker-compose.yml
      init/      (optional; initiale SQLs)
      sql/       (Schema/Migrationen)
    agents/      (deine Sammler/Agenten)
    ingest/      (Python-DB-Hilfen, z. B. db_client.py)

SCHNELLSTART
1) In den ad-db-Ordner wechseln:
   PowerShell:
     Set-Location "<Pfad>\ad-db"

2) Stack starten:
     docker compose up -d

3) Status prüfen:
     docker compose ps

Ergebnis:
- PostgreSQL läuft als Container "ad_pg" auf localhost:5432
- Adminer ist auf http://localhost:8080 erreichbar

ZUGANGSDATEN & VERBINDUNGS-URIS
- Datenbank: appdb
- Benutzer / Passwort: app / app
- Port: 5432
- Host:
  - vom Windows-Host / lokalen Skripten: localhost
  - von Containern im Compose-Netz: db

Beispiele:
- Python/psycopg URI: postgresql://app:app@localhost:5432/appdb
- Node.js (pg) URI:   postgres://app:app@localhost:5432/appdb

DB AUS DEM TERMINAL ÖFFNEN (PSQL)

Option A — psql im Container (funktioniert immer):
  Interaktiv:
    docker exec -it ad_pg psql -U app -d appdb
  Beenden: \q (Enter)
  Tabellenliste: \dt+

Option B — psql lokal (falls installiert):
  Interaktiv:
    psql -h localhost -U app -d appdb
  Einmaliger Befehl:
    psql -h localhost -U app -d appdb -c "SELECT now();"

  psql installieren (falls nötig):
    winget install PostgreSQL.PostgreSQL
  (neues Terminal öffnen, damit psql im PATH ist)

ADMINER (WEB-UI)
- Browser: http://localhost:8080
- Login:
  System:   PostgreSQL
  Server:   db
  Benutzer: app
  Passwort: app
  Datenbank: appdb

UMGEBUNGSVARIABLE FÜR SKRIPTE
PowerShell:
  $env:DATABASE_URL = "postgresql://app:app@localhost:5432/appdb"

Python:
  import os, psycopg
  conn = psycopg.connect(os.getenv("DATABASE_URL"))

SCHEMA-ÜBERBLICK (Kurz)
- campaigns — Kampagnen (id, name, slug, created_at)
- ads — Anzeigen je Kampagne (unique: campaign_id + ad_external_id; first_seen/last_seen)
- api_snapshots — Tages-JSON pro Kampagne (payload JSONB; unique: campaign_id + snapshot_date)
- ad_snapshots — Tages-JSON pro Ad (optional; payload JSONB)
- media — Medien je Ad (kind: image|video|screenshot; filename, file_url, date_folder, sha256, …)
- analyses — Analyse-Definitionen (name, provider, version, parameters JSONB)
- analysis_results — Analyse-Ergebnisse (target_type: ad|campaign; result JSONB, score)
- ensemble_links — Verknüpft Ensemble-Resultat mit OpenAI/Gemini-Quellen
- Views:
  - campaign_overview — Überblick pro Kampagne (Counts, neuestes Snapshot-Datum)
  - latest_llm_per_ad — je Ad das neueste Ergebnis pro Analyse & Provider

TYPISCHE WORKFLOWS (3 Sammler)

1) API-Sammler → api_snapshots
  SQL-Pattern:
    INSERT INTO api_snapshots(campaign_id, snapshot_date, payload, file_url)
    VALUES ($1,$2,$3,$4)
    ON CONFLICT (campaign_id, snapshot_date) DO UPDATE
      SET payload = EXCLUDED.payload,
          file_url = COALESCE(EXCLUDED.file_url, api_snapshots.file_url);

  Hinweis: In Python dict direkt an psycopg übergeben (wird zu JSONB konvertiert).

2) Screenshot-Sammler → ads + media(kind='screenshot')
  - Für jede Datei: <Screenshots/YYYY-MM-DD/Ad_ID.png>
  - ads upserten (unique (campaign_id, ad_external_id); last_seen aktualisieren)
  - media-Zeile anlegen (kind='screenshot', file_url='file:///...', filename, date_folder, sha256)
  - Idempotent dank ON CONFLICT und sha256.

3) Media-Sammler (images/videos) → ads + media
  - Analog Screenshots, mit kind='image' bzw. 'video'.

NÜTZLICHE PRÜF-QUERIES

-- Überblick / Dashboard
SELECT * FROM campaign_overview
ORDER BY latest_snapshot DESC NULLS LAST
LIMIT 10;

-- Aktuellste Medien einer Kampagne
SELECT m.kind, m.filename, m.date_folder, a.ad_external_id
FROM media m
JOIN ads a ON a.id = m.ad_id
JOIN campaigns c ON c.id = a.campaign_id
WHERE c.slug = 'sommer_2025'
ORDER BY m.date_folder DESC, m.created_at DESC
LIMIT 50;

-- Anzahl Ads je Kampagne
SELECT c.name, COUNT(*) AS ads
FROM ads a JOIN campaigns c ON c.id=a.campaign_id
GROUP BY c.name
ORDER BY ads DESC;

-- Größe eines API-Snapshots
SELECT c.name, a.snapshot_date, jsonb_array_length(a.payload) AS ads
FROM api_snapshots a
JOIN campaigns c ON c.id=a.campaign_id
ORDER BY a.id DESC
LIMIT 5;

BACKUPS & RESTORE

Backup (Dump) erstellen:
  docker exec -i ad_pg pg_dump -U app -d appdb | gzip > backup_$(Get-Date -Format "yyyy-MM-dd_HH-mm").sql.gz

Wiederherstellen:
  gunzip -c .\backup_YYYY-MM-DD_HH-mm.sql.gz | docker exec -i ad_pg psql -U app -d appdb

SCHEMA-UPDATES (MIGRATIONEN)

Neue SQL-Datei ausführen:
  docker cp .\sql\002_add_indexes.sql ad_pg:/tmp/002_add_indexes.sql
  docker exec ad_pg psql -U app -d appdb -f /tmp/002_add_indexes.sql

TROUBLESHOOTING

Container/Ports:
  docker compose ps
  docker compose logs db -f

Verbindungstest:
  docker exec -it ad_pg psql -U app -d appdb -c "SELECT 1;"

Schema geladen?
  docker exec ad_pg psql -U app -d appdb -c "\dt+"
  docker exec ad_pg psql -U app -d appdb -c "\dv+"

Windows-Fallen:
- PowerShell Here-Strings: für SQL immer einfachen Here-String nutzen @' ... '@ (sonst zerschießt es $$).
- Pfade mit Leerzeichen: in Anführungszeichen setzen.
- Containername: Befehle erwarten "ad_pg" (ggf. mit docker ps prüfen).

Mini-Smoke-Test:
  docker exec -i ad_pg psql -U app -d appdb -v ON_ERROR_STOP=1 -c "
  INSERT INTO campaigns(name, slug) VALUES ('Testkampagne','testkampagne') ON CONFLICT DO NOTHING;
  WITH c AS (SELECT id FROM campaigns WHERE slug='testkampagne')
  INSERT INTO ads(campaign_id, ad_external_id, first_seen, last_seen)
  SELECT c.id,'AD_123',now(),now() FROM c
  ON CONFLICT (campaign_id, ad_external_id) DO UPDATE SET last_seen=EXCLUDED.last_seen;
  "
  docker exec ad_pg psql -U app -d appdb -c "
  SELECT c.name, a.ad_external_id FROM campaigns c JOIN ads a ON a.campaign_id=c.id WHERE c.slug='testkampagne';
  "