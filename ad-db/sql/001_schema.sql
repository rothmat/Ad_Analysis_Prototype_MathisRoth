-- === Types ===
DO $$ BEGIN
  CREATE TYPE media_kind AS ENUM ('image','video','screenshot','other');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE target_kind AS ENUM ('campaign','ad');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE provider_kind AS ENUM ('openai','gemini','ensemble','other');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- === Core entities ===
CREATE TABLE IF NOT EXISTS campaigns (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ads (
  id BIGSERIAL PRIMARY KEY,
  campaign_id BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  ad_external_id TEXT NOT NULL,
  first_seen TIMESTAMPTZ,
  last_seen TIMESTAMPTZ,
  UNIQUE (campaign_id, ad_external_id)
);

CREATE TABLE IF NOT EXISTS api_snapshots (
  id BIGSERIAL PRIMARY KEY,
  campaign_id BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  snapshot_date DATE NOT NULL,
  payload JSONB NOT NULL,
  file_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (campaign_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS ad_snapshots (
  id BIGSERIAL PRIMARY KEY,
  ad_id BIGINT NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
  snapshot_date DATE NOT NULL,
  payload JSONB NOT NULL,
  file_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (ad_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS media (
  id BIGSERIAL PRIMARY KEY,
  ad_id BIGINT NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
  kind media_kind NOT NULL,
  format TEXT,
  file_url TEXT NOT NULL,
  filename TEXT,
  date_folder DATE,
  bytes BIGINT,
  width INTEGER,
  height INTEGER,
  duration_seconds DOUBLE PRECISION,
  sha256 CHAR(64),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_media_ad ON media(ad_id);
CREATE INDEX IF NOT EXISTS idx_media_sha ON media(sha256);

CREATE TABLE IF NOT EXISTS analyses (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  provider provider_kind NOT NULL,
  version TEXT,
  parameters JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name, provider, version)
);

CREATE TABLE IF NOT EXISTS analysis_results (
  id BIGSERIAL PRIMARY KEY,
  analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  target_type target_kind NOT NULL,
  campaign_id BIGINT REFERENCES campaigns(id) ON DELETE CASCADE,
  ad_id BIGINT REFERENCES ads(id) ON DELETE CASCADE,
  result JSONB NOT NULL,
  score DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT one_target CHECK (
    (target_type = 'campaign' AND campaign_id IS NOT NULL AND ad_id IS NULL) OR
    (target_type = 'ad'       AND ad_id IS NOT NULL       AND campaign_id IS NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_results_analysis ON analysis_results(analysis_id);
CREATE INDEX IF NOT EXISTS idx_results_campaign ON analysis_results(campaign_id);
CREATE INDEX IF NOT EXISTS idx_results_ad ON analysis_results(ad_id);
CREATE INDEX IF NOT EXISTS idx_results_json_gin ON analysis_results USING GIN (result jsonb_path_ops);

CREATE TABLE IF NOT EXISTS ensemble_links (
  ensemble_result_id BIGINT PRIMARY KEY REFERENCES analysis_results(id) ON DELETE CASCADE,
  source_openai_result_id BIGINT REFERENCES analysis_results(id) ON DELETE SET NULL,
  source_gemini_result_id BIGINT REFERENCES analysis_results(id) ON DELETE SET NULL
);

CREATE OR REPLACE VIEW latest_llm_per_ad AS
SELECT DISTINCT ON (ar.ad_id, a.provider, a.name)
  ar.*
FROM analysis_results ar
JOIN analyses a ON a.id = ar.analysis_id
WHERE ar.target_type = 'ad'
ORDER BY ar.ad_id, a.provider, a.name, ar.created_at DESC;

CREATE OR REPLACE VIEW campaign_overview AS
SELECT
  c.id AS campaign_id,
  c.name,
  COUNT(DISTINCT ad.id) AS ads,
  COUNT(DISTINCT m.id) FILTER (WHERE m.kind='image') AS images,
  COUNT(DISTINCT m.id) FILTER (WHERE m.kind='video') AS videos,
  COUNT(DISTINCT m.id) FILTER (WHERE m.kind='screenshot') AS screenshots,
  MAX(s.snapshot_date) AS latest_snapshot
FROM campaigns c
LEFT JOIN ads ad ON ad.campaign_id = c.id
LEFT JOIN media m ON m.ad_id = ad.id
LEFT JOIN api_snapshots s ON s.campaign_id = c.id
GROUP BY c.id, c.name;

CREATE TABLE IF NOT EXISTS ad_llm_fused (
  id             BIGSERIAL PRIMARY KEY,
  ad_id          BIGINT NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
  snapshot_date  DATE   NOT NULL,
  fused          JSONB  NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (ad_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_ad_llm_fused_ad_id_date
  ON ad_llm_fused (ad_id, snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_ad_llm_fused_gin
  ON ad_llm_fused USING GIN (fused jsonb_path_ops);

CREATE TABLE IF NOT EXISTS ad_topics_results (
  id BIGSERIAL PRIMARY KEY,
  ad_id BIGINT NOT NULL,
  page_name TEXT,
  campaign_slug TEXT,
  topics JSONB,
  rationale_bullets JSONB,
  model TEXT,
  analyzed_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (ad_id, model)
);

CREATE TABLE IF NOT EXISTS ad_weaknesses_results (
  id BIGSERIAL PRIMARY KEY,
  ad_id BIGINT NOT NULL,
  overall_risk DOUBLE PRECISION,
  overall_confidence DOUBLE PRECISION,
  categories JSONB,
  notes TEXT,
  model TEXT,
  analyzed_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (ad_id, model)
);

CREATE TABLE IF NOT EXISTS campaign_perspective_results (
    id BIGSERIAL PRIMARY KEY,
    campaign_slug TEXT NOT NULL,
    page_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    stance TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    rationale_bullets JSONB,
    model TEXT,
    ad_ids JSONB,
    analyzed_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (campaign_slug, page_name, topic)
);