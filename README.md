# GNNWR Platform Backend

FastAPI backend for the **GNNWR / GTNNWR** spatiotemporal analysis cloud platform.

This repository is the dedicated **backend service** (companion to independent frontend [`gnnwr_platform_frontend`](https://github.com/llzays-eng/gnnwr_platform_frontend)). It ports and upgrades the public monorepo [`gnnwr_platform`](https://github.com/llzays-eng/gnnwr_platform) `backend/` + `engine/` into a clean, runnable service at the repo root.

Product flow: **upload → preprocess → async train → WebSocket progress → results / coefficients / compare → spatial tiles / surface → report export**. Scenarios: housing prices (GNNWR) and air quality (GTNNWR).

## Frozen integration notes (frontend)

| Item | This backend |
|---|---|
| REST prefix | `/api/v1` |
| Login | `POST /api/v1/auth/login` — **JSON** `{username, password}` *or* OAuth2 password form (`username` = email or username) |
| **WebSocket path** | **`/ws/models/tasks/{task_id}`** (legacy alias: `/api/v1/models/tasks/{task_id}/ws`) |
| WS auth | Query **`?token=`** (JWT access token) and/or **`?ticket=`** from `POST /api/v1/auth/ws-ticket`. Unauthenticated connections are **rejected when not in DEBUG** (`WS_REQUIRE_AUTH=true`). Send text `ping` → `{type: pong}` heartbeat. |
| WS payload | `{ "type": "progress", "task_id", "status", "progress": { epoch, total_epochs, train_loss, val_loss, elapsed_s, eta_s }, "coef_summary"? }` plus `{type: status}` / `{type: log}` / `{type: error}` / `{type: pong}` |
| **`tile_crs`** | `GET /api/v1/spatial/surface/{task_id}` **always includes `tile_crs`**. Default **`GCJ02`**: this service’s XYZ/WMS treats tile lon/lat as GCJ-02 (route A — truthful, matches Amap). Not a GeoServer-only stub. **WMS `TIME` is not supported** (`wms_time_supported: false`; `time_dimension` may still list GTNNWR time keys for the vector layer). |
| Vector tiles | `GET /api/v1/spatial/tiles` stores WGS84; **default output `crs: WGS84`** so the frontend `toRenderCRS()` does not double-shift. Pass `output_crs=GCJ02` if the client will *not* transform. |
| Task list | **`GET /api/v1/models/tasks?project_id=`** (Paginated) — this was missing in the monorepo |
| Coefficients | Full dump by default (`limit=0`); optional `bbox`, `cursor`, `limit`, `vars` |

Lists return `{ items, total, page, page_size }` (matches frontend `Paginated<T>`), not bare arrays.

## Layout

```
app/                 # FastAPI 空间服务层
  api/v1/            # 路由（薄）
  schemas/           # Pydantic 契约
  models/            # SQLAlchemy ORM
  services/          # 存储 / CRS / 入库 / 曲面渲染 / 引擎切换
  tasks/             # Celery：cpu_queue 清洗、gpu_queue 训练
  ws/                # /ws/models/tasks/{task_id}
engine/              # 纯计算，无 FastAPI 依赖
  gnnwr_lite.py      # 默认 numpy 引擎
  gnnwr_torch.py     # 官方 gnnwr+torch 替换点（未装则勿开启）
  pipeline.py        # run_analysis 契约
deploy/postgres/     # init.sql（PostGIS + 表 + GiST）
docker-compose.yml
```

CRS rule: **compute & store WGS84**; CRS math lives only in `app/services/crs_transform.py`. Engine swap: `ENGINE_BACKEND=gnnwr_lite` (default) or `gnnwr_torch`.

## Quick start

### Docker Compose

```bash
cp .env.example .env          # 生产务必改 SECRET_KEY
docker compose up -d --build
# API     http://localhost:8000/docs
# MinIO   http://localhost:9001
# GeoServer（可选）: docker compose --profile extras up -d
```

Services: PostGIS, Redis, MinIO, API, `worker-cpu` (`cpu_queue`), `worker-gpu` (`gpu_queue`; runs `gnnwr_lite` on CPU unless you install torch). API and workers share a storage volume (and MinIO).

### Local development

Need PostGIS + Redis (or `docker compose up -d postgis redis minio`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env 中 POSTGRES_HOST/REDIS_HOST 指向本机或 compose 端口
uvicorn app.main:app --reload --port 8000
celery -A app.tasks.celery_app.celery_app worker -Q cpu_queue,gpu_queue -l info
```

If Celery/Redis is down, preprocess and train **fall back to in-process execution** so `/docs` experiments still work.

OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Tests

```bash
pytest                       # 引擎冒烟 + OpenAPI 路径 + CRS + 契约
python -m engine.test_engine # 完整引擎回归（更慢，含精度断言）
```

## Auth & main routes

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/auth/register` | |
| POST | `/api/v1/auth/login` | JSON or OAuth2 form |
| GET | `/api/v1/auth/me` | JWT |
| POST | `/api/v1/auth/refresh` | body `{refresh_token}` |
| POST | `/api/v1/auth/logout` | access `jti` blacklist (Redis) |
| POST | `/api/v1/auth/ws-ticket` | 120s ticket for WS |
| POST/GET | `/api/v1/projects` | create; list Paginated |
| GET/DELETE | `/api/v1/projects/{id}` | |
| POST | `/api/v1/datasets/upload` | multipart `project_id`,`file`,`source_crs?` · 200MB |
| POST | `/api/v1/datasets/upload/init` → PUT `.../chunk/{n}` → POST `.../complete` | chunked upload |
| POST | `/api/v1/datasets/upload/presign` | MinIO PUT URL when MinIO is up |
| GET | `/api/v1/datasets?project_id=` | Paginated, `project_id` required |
| GET | `/api/v1/datasets/{id}/preview` | `spatial_sample` + GCJ-02 `points` + `mapping_suggestion` |
| POST | `/api/v1/datasets/{id}/preprocess` | 202, Celery `cpu_queue`; status `uploaded→cleaning→cleaned→ingested` |
| POST | `/api/v1/models/train` | 202 |
| GET | `/api/v1/models/tasks?project_id=` | **P0 list** |
| GET | `/api/v1/models/tasks/{id}/status\|result\|compare\|coefficients` | |
| POST | `/api/v1/models/tasks/{id}/cancel` | mark + best-effort revoke |
| GET | `/api/v1/models/tasks/{id}/logs?since=` | |
| GET | `/api/v1/spatial/tiles` | `bbox` required; `fields`, `cursor`, time filters |
| GET | `/api/v1/spatial/surface/{id}` | **`tile_crs` required in body** |
| GET | `/api/v1/reports/{id}/export` | sync PDF or HTML |

Business routes require `Authorization: Bearer <access_token>`.

## Known gaps vs frontend `PENDING`

Implemented here so the frontend can freeze them: project/dataset/task lists, chunked upload, coefficients bbox paging, cancel/logs, logout/refresh, `tile_crs`, WS path + token.

Still not in this baseline:

- Config templates (`/config-templates`) — frontend localStorage is fine
- Async report job (`GET /reports/jobs/{id}`) — export stays **sync** file/HTML; frontend already probes blob vs JSON
- GeoServer JWT gateway proxy — tiles are rendered by this API (no GeoServer secret in the browser). Optional GeoServer remains a compose profile
- Official `gnnwr+torch` adapter body in `engine/gnnwr_torch.py` (swap point + error if enabled without install)
- Shapefile `.zip` parser (upload types CSV / Excel / GeoJSON)

## Environment

See `.env.example`. Important: `SECRET_KEY`, `SURFACE_TILE_CRS=GCJ02`, `VECTOR_OUTPUT_CRS=WGS84`, `ENGINE_BACKEND=gnnwr_lite`, `WS_REQUIRE_AUTH`.
