-- GNNWR 平台数据库初始化
-- 容器首次启动时自动执行：PostGIS 扩展 + 核心表 + 空间索引。
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email VARCHAR(200) UNIQUE NOT NULL,
  username VARCHAR(100) NOT NULL,
  display_name VARCHAR(100) DEFAULT '',
  hashed_password VARCHAR(255) NOT NULL,
  role VARCHAR(20) DEFAULT 'user',
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS projects (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(200) NOT NULL,
  description VARCHAR(500) DEFAULT '',
  scenario_type VARCHAR(50) DEFAULT 'custom',
  is_demo BOOLEAN DEFAULT false,
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS datasets (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  name VARCHAR(200) DEFAULT '',
  filename VARCHAR(200) DEFAULT '',
  storage_path VARCHAR(400) NOT NULL,
  file_type VARCHAR(20),
  format VARCHAR(20) DEFAULT 'csv',
  source_crs VARCHAR(30),
  size_bytes INT DEFAULT 0,
  row_count INT DEFAULT 0,
  status VARCHAR(20) DEFAULT 'uploaded',
  status_detail JSONB,
  schema JSONB DEFAULT '[]'::jsonb,
  column_guess JSONB DEFAULT '{}'::jsonb,
  mapping JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS spatial_features (
  id BIGSERIAL PRIMARY KEY,
  dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
  geom GEOMETRY(Point, 4326),
  observed_time TIMESTAMP,
  properties JSONB
);
CREATE INDEX IF NOT EXISTS idx_spatial_features_geom
  ON spatial_features USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_spatial_features_dataset
  ON spatial_features(dataset_id);
CREATE INDEX IF NOT EXISTS idx_spatial_features_time
  ON spatial_features(observed_time);

CREATE TABLE IF NOT EXISTS model_tasks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
  model_type VARCHAR(10),
  x_columns JSONB,
  y_column VARCHAR(100),
  spatial_columns JSONB,
  temporal_column VARCHAR(100),
  hyperparams JSONB,
  celery_task_id VARCHAR(100),
  status VARCHAR(20) DEFAULT 'PENDING',
  progress DOUBLE PRECISION DEFAULT 0,
  progress_detail JSONB,
  error VARCHAR(500),
  error_code VARCHAR(64),
  error_retryable BOOLEAN,
  created_at TIMESTAMP DEFAULT now(),
  started_at TIMESTAMP,
  finished_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_results (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  task_id UUID REFERENCES model_tasks(id) ON DELETE CASCADE,
  r2 FLOAT, rmse FLOAT, mae FLOAT, aicc FLOAT,
  model_weight_path VARCHAR(300),
  coefficients_summary JSONB,
  residuals_path VARCHAR(300),
  sample_count INT,
  loss_history JSONB DEFAULT '[]'::jsonb,
  residual_summary JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS baseline_comparisons (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  task_id UUID REFERENCES model_tasks(id) ON DELETE CASCADE,
  method VARCHAR(20),
  r2 FLOAT, rmse FLOAT, mae FLOAT, aicc FLOAT,
  coefficients_summary JSONB
);
