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
| WS auth | Query **`?token=`** (JWT access token) and/or **`?ticket=`** from `POST /api/v1/auth/ws-ticket` (optional `task_id` bind). Tickets are **one-shot** (Redis `GETDEL` on connect). Unauthenticated connections are **rejected when not in DEBUG** (`WS_REQUIRE_AUTH=true`). Send text `ping` → `{type: pong}` heartbeat. |
| WS payload | `{ "type": "progress", "task_id", "status", "progress": { epoch, total_epochs, train_loss, val_loss, elapsed_s, eta_s }, "coef_summary"? }` plus `{type: status}` / `{type: log}` / `{type: error}` / `{type: pong}`. `status` is only `PENDING\|RUNNING\|SUCCESS\|FAILED` (cancel is **FAILED** + `error.code=TASK_CANCELLED`). |
| **`tile_crs`** | `GET /api/v1/spatial/surface/{task_id}` **always includes `tile_crs`**. Default **`GCJ02`**: this service’s XYZ/WMS treats tile lon/lat as GCJ-02 (route A — truthful, matches Amap). WMS GetCapabilities CRS matches `SURFACE_TILE_CRS` (GCJ02 is **not** advertised as `CRS:84`). **WMS `TIME` is not supported** (`wms_time_supported: false`; `time_dimension` may still list GTNNWR time keys for the vector layer). |
| Surface img/WMS auth | Map `<img>` and WMS cannot send `Authorization`. JWT surface metadata returns XYZ / WMS / legend URLs **already including** a short-lived query `sig` (`typ=tile`, bound to user + `task_id`, TTL `TILE_TOKEN_EXPIRE_MINUTES`). Missing / invalid / expired / wrong-task signatures are **401**. A task UUID in the path is **not** authorization. |
| Vector tiles | `GET /api/v1/spatial/tiles` requires owning the dataset’s project (else **404**). Store is WGS84; **default output `crs: WGS84`**. `bbox_crs` defaults to **WGS84** (frontend `BBox`); only `bbox_crs=GCJ02` transforms corners. Pass `output_crs=GCJ02` if the client will *not* transform. |
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

If Celery/Redis is down, train/preprocess return **503** unless you explicitly set `ALLOW_INLINE_JOBS=true` (dev only). The API process does **not** silently run training inline.

OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Auth threat model (tiles / WS)

- **Knowing a UUID is not auth.** `dataset_id` / `task_id` in the path only identify the resource; ownership is checked via JWT, or via a signed tile `sig`, or a one-shot WS ticket.
- **Browser map tiles** (`<img>`, WMS `GetMap`) cannot attach `Authorization`. Those three routes (`/spatial/surface/{id}/xyz/...png`, `/legend.png`, `/spatial/wms/{id}`) require query `sig`. The JWT `GET /spatial/surface/{id}` mints that signature so the frontend can paste URLs into Amap.
- Tile `sig` is a JWT (`typ=tile`) bound to `sub` (user) + `tid` (task), reusable until TTL (default 60 minutes) so the map can fetch many tiles. It is **not** an access token and is rejected on other routes.
- **WS tickets** are one-shot: `POST /auth/ws-ticket` stores `{user_id, task_id?}` in Redis; the first successful `?ticket=` consume (`GETDEL`) deletes it. Replay is denied. Prefer `?token=` for reconnects.
- **`SECRET_KEY`**: default `DEBUG=false`. If `SECRET_KEY` is still the `CHANGE_ME…` placeholder, the process **refuses to start** when `DEBUG` is false; DEBUG-only logs a loud warning.

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
| POST | `/api/v1/auth/ws-ticket` | 120s **one-shot** ticket (`task_id` optional bind) |
| POST/GET | `/api/v1/projects` | create; list Paginated |
| GET/DELETE | `/api/v1/projects/{id}` | |
| POST | `/api/v1/datasets/upload` | multipart `project_id`,`file`,`source_crs?` · 200MB |
| POST | `/api/v1/datasets/upload/init` → PUT `.../chunk/{n}` → POST `.../complete` | chunked upload |
| POST | `/api/v1/datasets/upload/presign` | MinIO PUT URL when MinIO is up |
| GET | `/api/v1/datasets?project_id=` | Paginated, `project_id` required |
| GET | `/api/v1/datasets/{id}/preview` | `spatial_sample` + GCJ-02 `points` + `mapping_suggestion` |
| POST | `/api/v1/datasets/{id}/preprocess` | 202, Celery `cpu_queue`; status `uploaded→cleaning→cleaned→ingested` |
| POST | `/api/v1/models/train` | 202; **503** if broker down unless `ALLOW_INLINE_JOBS` |
| GET | `/api/v1/models/tasks?project_id=` | **P0 list** |
| GET | `/api/v1/models/tasks/{id}/status\|result\|compare\|coefficients` | |
| POST | `/api/v1/models/tasks/{id}/cancel` | public status **FAILED**, `error.code=TASK_CANCELLED`, `retryable=false` |
| GET | `/api/v1/models/tasks/{id}/logs?since=` | |
| GET | `/api/v1/spatial/tiles` | owner check; `bbox`; `bbox_crs=WGS84` default; `fields`, `cursor`, time filters |
| GET | `/api/v1/spatial/surface/{id}` | JWT; **`tile_crs`**; URLs include `sig` |
| GET | `/api/v1/spatial/surface/{id}/xyz/{z}/{x}/{y}.png` | **`sig` required** |
| GET | `/api/v1/spatial/surface/{id}/legend.png` | **`sig` required** |
| GET | `/api/v1/spatial/wms/{id}` | **`sig` required**; GetCapabilities CRS = `SURFACE_TILE_CRS` |
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

See `.env.example`. Important: `SECRET_KEY` (must not be the placeholder when `DEBUG=false`), `DEBUG=false` default, `ALLOW_INLINE_JOBS=false`, `TILE_TOKEN_EXPIRE_MINUTES`, `SURFACE_TILE_CRS=GCJ02`, `VECTOR_OUTPUT_CRS=WGS84`, `ENGINE_BACKEND=gnnwr_lite`, `WS_REQUIRE_AUTH`.
