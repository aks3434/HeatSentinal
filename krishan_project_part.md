# 🗄️ HeatSentinel — Database & PostGIS Integration Guide (Krishan's Module)

> **Assigned Developer**: Krishan  
> **Module**: PostgreSQL 15 + PostGIS Spatial Database & Data Persistence  
> **Status of Brain/ML/API (Team Lead)**: ✅ 100% Complete & Running on `http://127.0.0.1:8000`

---

## 🛑 1. Git & Conflict-Prevention Rules (CRITICAL)

To prevent merge and stack conflicts with the completed ML Brain and FastAPI endpoints, follow these boundary rules:

1. **Your Workspace (Files you own and edit)**:
   - `backend/app/db/database.py` (SQLAlchemy engine & session management)
   - `backend/app/db/models.py` (ORM table definitions)
   - `backend/app/db/session.py` (FastAPI `get_db` dependency injection)
   - `docker-compose.yml` (PostgreSQL + PostGIS container configuration)
   - `alembic/` (If you configure database migrations)

2. **Protected Files (DO NOT modify or overwrite)**:
   - ❌ `ml_engine/` — All ML models (XGBoost, Bi-LSTM, Stacking Ensemble, SHAP) are trained and finalized.
   - ❌ `slm_engine/` — Fine-tuning corpus, IndicTrans2 translation, and inference engine are finalized.
   - ❌ `data/` — Contains Parquet datasets, GeoJSON district boundaries, and trained model weights (`.pt`, `.json`).
   - ❌ `backend/app/api/v1/predict.py`, `explain.py`, `chat.py` — Lead's core endpoints.

---

## 🎯 2. What Krishan Needs to Build

Your role is to build the persistence layer that stores:
1. **Geospatial District Polygons** (using PostGIS geometry so spatial queries can be executed).
2. **Registered Citizens/Users** (storing phone numbers, pincodes, and Point coordinates for alert dispatching).
3. **Historical Prediction Logs** (caching daily predictions made by the ML engine).
4. **Alert Dispatch Logs** (recording which users received SMS/Push alerts to prevent spam).

---

## 📐 3. Database Schema Specification (`backend/app/db/models.py`)

Implement the following SQLAlchemy models (with `GeoAlchemy2` for spatial types):

### Table 1: `districts` (Geospatial Table)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `Integer`, Primary Key | Unique internal ID |
| `district_code` | `String(10)`, Unique, Indexed | E.g. `"RJ01"`, `"MH02"` |
| `name` | `String(100)` | District Name (e.g. `"Jaipur"`) |
| `state` | `String(100)` | State Name (e.g. `"Rajasthan"`) |
| `baseline_normal`| `Float` | Normal summer baseline max temp (°C) |
| `geom` | `Geometry(geometry_type='MULTIPOLYGON', srid=4326)` | PostGIS boundary polygon |

### Table 2: `users` (Citizen Subscriptions)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `Integer`, Primary Key | User ID |
| `phone_number` | `String(15)`, Unique | E.g. `"+919876543210"` |
| `pincode` | `String(6)` | Indian 6-digit postal code |
| `preferred_lang`| `String(5)` | `"hi"`, `"mr"`, `"te"`, `"ta"`, `"bn"`, or `"en"` |
| `location` | `Geometry(geometry_type='POINT', srid=4326)` | User GPS/centroid location |
| `created_at` | `DateTime` | Registration timestamp |

### Table 3: `prediction_logs` (Cached Daily Forecasts)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `Integer`, Primary Key | Log ID |
| `district_code` | `String(10)`, ForeignKey(`districts.district_code`) | District identifier |
| `forecast_date` | `Date` | Prediction date |
| `predicted_max_temp`| `Float` | Model predicted max temperature |
| `heatwave_prob`| `Float` | Calibrated stacking ensemble probability ($P_{\text{final}}$) |
| `severity` | `String(20)` | `"Normal"`, `"Moderate"`, `"Severe"`, `"Extreme"` |
| `hvi_score` | `Float` | Heat Vulnerability Index |
| `expected_heat_strokes`| `Integer` | Epidemiological health impact prediction |

### Table 4: `alert_logs` (Spam Prevention Log)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `Integer`, Primary Key | Alert Log ID |
| `user_id` | `Integer`, ForeignKey(`users.id`) | Recipient citizen |
| `district_code` | `String(10)` | District that triggered the alert |
| `channel` | `String(10)` | `"SMS"`, `"PUSH"`, or `"WEBHOOK"` |
| `dispatched_at`| `DateTime` | Timestamp of alert |
| `status` | `String(20)` | `"SENT"`, `"FAILED"`, `"DELIVERED"` |

---

## 🌐 4. PostGIS Spatial Query Requirement

One standout feature for our project presentation is executing a spatial query in PostgreSQL to find which citizens fall inside an affected district:

```sql
-- Spatial query for the Alert System: Find all users inside an affected district
SELECT u.phone_number, u.preferred_lang 
FROM users u, districts d
WHERE d.district_code = :affected_district_code
  AND ST_Contains(d.geom, u.location);
