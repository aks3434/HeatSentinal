# 🏗️ HeatSentinel — Low-Level Deep-Dive System Architecture

---

## 1. 🌐 System Overview & Dataflow Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EXTERNAL DATA SOURCES                                  │
│  IMD APIs / CSVs │ NASA POWER (Solar/Wind) │ OpenMeteo ERA5 │ ISRO Bhuvan (NDVI) │ POSOCO    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                  ┌─────────▼──────────┐
                                  │   Airflow / Cron   │ (Daily ETL at 02:00 AM)
                                  │   Data Pipeline    │
                                  └─────────┬──────────┘
                                            │ Parquet Storage
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
          ┌─────────▼──────────┐                         ┌──────────▼───────────┐
          │  XGBoost Engine    │                         │  Bi-LSTM Engine      │
          │ (Tabular Features) │                         │ (14-Day Sequence)    │
          └─────────┬──────────┘                         └──────────┬───────────┘
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            │ Model Stacking
                                  ┌─────────▼──────────┐
                                  │ Meta-Learner Model │ ───► SHAP Explainer Engine
                                  │ (Probability + Temp)│
                                  └─────────┬──────────┘
                                            │
                                  ┌─────────▼──────────┐
                                  │  FastAPI Backend   │
                                  └────┬───────────┬───┘
                                       │           │
           ┌───────────────────────────┘           └──────────────────────────┐
           │ Async Job Queue                                                  │ API Endpoints
 ┌─────────▼──────────┐                                             ┌──────────▼───────────┐
 │   Celery + Redis   │                                             │    React Frontend    │
 └─────────┬──────────┘                                             │ (Leaflet + Recharts) │
           │                                                        └──────────┬───────────┘
 ┌─────────▼──────────┐                                                        │
 │ PostGIS Location   │                                             ┌──────────▼───────────┐
 │ Polygon Query      │                                             │   Fine-Tuned SLM     │
 └─────────┬──────────┘                                             │   (Phi-3 / Groq)     │
           │                                                        └──────────────────────┘
 ┌─────────▼──────────┐
 │ Alert Dispatcher   │
 │ (SMS / Push FCM)   │
 └────────────────────┘
```

---

## 2. 📊 Layer 1: Data Pipeline & Feature Engineering

### 2.1 Ingestion Sources
- **IMD (India Meteorological Department)**: Historical & daily maximum/minimum temperatures, humidity, station reports.
- **NASA POWER API**: Surface solar radiation ($W/m^2$), 2m wind speed ($m/s$), dew point ($^\circ C$).
- **OpenMeteo ERA5 Reanalysis**: 7-day forecast data for grid coordinates across India.
- **ISRO Bhuvan / MODIS**: NDVI (Normalized Difference Vegetation Index) — identifies drought & vegetation deficit.
- **Census of India**: District-level demographic data (elderly ratio, outdoor worker population density, slum density).

### 2.2 Feature Engineering Formulas & Pipeline
From raw temperature ($T$ in $^\circ C$) and relative humidity ($RH$ in $\%$):

1. **Heat Index (Steadman / Rothfusz Regression)**:
   $$HI = c_1 + c_2 T + c_3 RH + c_4 T \cdot RH + c_5 T^2 + c_6 RH^2 + c_7 T^2 RH + c_8 T RH^2 + c_9 T^2 RH^2$$
2. **Wet Bulb Globe Temperature (WBGT Approximation)**:
   $$WBGT \approx 0.567 T + 0.393 e + 3.94$$
   *(where $e$ is vapor pressure derived from humidity)*
3. **Temporal Ingestion Vectors**:
   - `rolling_mean_3d`, `rolling_mean_7d`, `rolling_max_14d` of daily max temperature.
   - `consecutive_hot_days`: Streak count of days where $T_{max} \ge 40^\circ C$.
   - `temp_anomaly`: $T_{max} - T_{normal\_historical\_mean}$.
4. **Spatial Vectors**:
   - `elevation_m`, `urban_built_up_ratio`, `ndvi_deficit`.

---

## 3. 🤖 Layer 2: Machine Learning & Explainability Engine

### 3.1 Dual-Model Stacking Ensemble
- **Model A: XGBoost Classifier & Regressor**
  - *Classifier*: Outputs `heatwave_probability` ($P \in [0, 1]$). Target label based on official IMD criteria: $T_{max} \ge 40^\circ C$ (plains) and anomaly $\ge 4.5^\circ C$ above normal for $\ge 2$ consecutive days.
  - *Regressor*: Predicts exact `predicted_max_temp` and `predicted_heat_index`.
- **Model B: Bidirectional LSTM (PyTorch)**
  - Input tensor shape: `(batch_size, seq_len=14, num_features=18)`
  - 2x Bi-LSTM layers (hidden_dim=128, dropout=0.2) + Fully Connected Linear Head.
  - Captures 14-day temporal momentum, pressure variations, and atmospheric inertia.
- **Meta-Learner (Stacking)**:
  - Logistic Regression meta-model combining XGBoost output probabilities and LSTM temporal sequence embeddings into a final calibrated risk score.

### 3.2 Explainable AI (SHAP Engine)
- Integrated `shap.TreeExplainer` on the XGBoost sub-model.
- For every prediction generated by the backend, the system calculates exact SHAP attribution values:
  $$\text{Prediction} = \text{Base Value} + \sum_{i=1}^{k} \text{SHAP}_i$$
- *Example output delivered via API*:
  - Base risk: 15%
  - `consecutive_hot_days = 5` $\rightarrow$ +32%
  - `humidity = 78%` $\rightarrow$ +24%
  - `wind_speed = 12 km/h` $\rightarrow$ -8%
  - **Final Risk Score: 63% (Severe Heatwave Alert)**

### 3.3 Heat Vulnerability Index (HVI) & Health Impact Model
- **Composite HVI Equation**:
  $$HVI = 0.40 \cdot \text{MeteorologicalRisk} + 0.25 \cdot \text{ElderlyPop\%} + 0.20 \cdot \text{SlumDensity} + 0.15 \cdot \text{OutdoorWorkers\%}$$
- **Health Impact Regression**:
  $$\text{ExpectedHeatStrokes} = f(HVI, \text{TotalPopulation}, \text{PredictedMaxTemp})$$

---

## 4. 🧠 Layer 3: Fine-Tuned Small Language Model (SLM) & Translation

### 4.1 Fine-Tuning Strategy (LoRA via HuggingFace PEFT)
- **Base Model**: `Phi-3-mini-4k-instruct` (3.8B parameters) or `Qwen2-1.5B-Instruct`.
- **LoRA Hyperparameters**:
  - Rank $r = 16$, $\alpha = 32$, Dropout $= 0.05$.
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`.
- **Training Corpus Structure**:
  - IMD Heatwave Action Plan PDFs (converted to structured text).
  - Synthetic Q&A dataset generated from actual prediction outputs (`district`, `temp`, `severity`, `recommended_actions`).
  - District-to-State geographic taxonomy pairs.
- **Serving Options**:
  - *Option A*: Local inference using `vLLM` / `llama.cpp` exposed via FastAPI.
  - *Option B*: Free Groq LLaMA-3-8B API with custom system prompts (fallback / fast demo mode).

### 4.2 Multilingual Engine (IndicTrans2)
- Integrates AI4Bharat's **IndicTrans2** model service.
- Converts generated English advisories into regional languages (**Hindi, Marathi, Telugu, Tamil, Bengali**) before dispatching via SMS.

---

## 5. ⚙️ Layer 4: FastAPI Backend & PostGIS Database Architecture

### 5.1 Relational & Geospatial Schema (PostgreSQL + PostGIS)

```sql
-- Districts GeoJSON / MultiPolygon Table
CREATE TABLE districts (
    district_code VARCHAR(10) PRIMARY KEY,
    district_name VARCHAR(100) NOT NULL,
    state_name VARCHAR(100) NOT NULL,
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL,
    baseline_normal_temp FLOAT NOT NULL
);

-- Users & Alert Subscriptions
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number VARCHAR(15) UNIQUE NOT NULL,
    pincode VARCHAR(6) NOT NULL,
    preferred_language VARCHAR(10) DEFAULT 'hi',
    location GEOMETRY(Point, 4326),
    district_code VARCHAR(10) REFERENCES districts(district_code),
    fcm_push_token TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Heatwave Predictions Storage
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    district_code VARCHAR(10) REFERENCES districts(district_code),
    prediction_date DATE NOT NULL,
    forecast_for_date DATE NOT NULL,
    max_temp FLOAT NOT NULL,
    heat_index FLOAT NOT NULL,
    probability FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL, -- Normal, Moderate, Severe, Extreme
    shap_breakdown JSONB NOT NULL,
    hvi_score FLOAT NOT NULL,
    expected_heat_stroke_cases INT
);

-- Dispatch Logs
CREATE TABLE alert_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    district_code VARCHAR(10) REFERENCES districts(district_code),
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    channel VARCHAR(10), -- SMS, FCM, EMAIL
    message_body TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'SENT'
);
```

### 5.2 Async Background Job Queue (Celery + Redis)
- **Schedule**: Cron trigger at **06:00 AM daily**.
- **Job Workflow**:
  1. Fetch latest OpenMeteo 7-day weather forecast.
  2. Run ML Stacking Model pipeline for all ~700 Indian districts.
  3. Insert predictions into `predictions` table.
  4. Filter districts where `severity IN ('Severe', 'Extreme')`.
  5. Run PostGIS spatial query to get all registered users in those districts:
     ```sql
     SELECT u.id, u.phone_number, u.preferred_language, u.fcm_push_token, d.district_name
     FROM users u
     JOIN districts d ON ST_Contains(d.geom, u.location)
     WHERE d.district_code = :target_district_code;
     ```
  6. Pass user list to IndicTrans2 $\rightarrow$ generate translated SMS $\rightarrow$ fire Twilio / MSG91 API & FCM Push.

### 5.3 FastAPI Endpoint Specifications
- `GET /api/v1/predict?district_code=RJ01` $\rightarrow$ Current & 7-day forecast + severity + HVI.
- `GET /api/v1/map/geojson` $\rightarrow$ Complete district polygons merged with latest risk levels for Leaflet styling.
- `GET /api/v1/explain?district_code=RJ01&date=2026-08-10` $\rightarrow$ SHAP waterfall JSON data.
- `POST /api/v1/chat` $\rightarrow$ Accepts user prompt + district context, streams SLM response.
- `POST /api/v1/alerts/register` $\rightarrow$ Registers new user with phone, pincode, and converts pincode to Lat/Long via Nominatim Geocoder into PostGIS `Point`.

---

## 6. 🎨 Layer 5: React Frontend Topology

### 6.1 UI Component Tree & State Flow
```
App (Vite + React)
 ├── Navbar (Theme Toggle, Language Selector, Emergency Banner)
 ├── Dashboard View
 │    ├── Interactive Map Component (Leaflet.js + GeoJSON district layer)
 │    ├── District Detail Drawer (Slides in on map click)
 │    │    ├── 7-Day Forecast Chart (Recharts AreaChart)
 │    │    ├── SHAP Explainability Card (Interactive Bar/Waterfall)
 │    │    ├── Health & Energy Impact Metrics
 │    │    └── Cooling Center Locator (OSM Overpass API fetch)
 ├── SLM Chatbot Drawer / Widget (Floating Chat UI connected to FastAPI SSE stream)
 ├── Alert Registration Modal (Phone + Pincode form)
 └── Admin Portal View (Passcode protected: system health, manual alert override, PDF download)
```

---

## 🔒 7. Reliability, Security & Fallback Strategies

| Layer | Potential Failure | Fallback / Redundancy Strategy |
|-------|-------------------|--------------------------------|
| **Data API** | NASA / IMD API downtime | Fall back to OpenMeteo ERA5 historical archive |
| **SLM Inference** | Local GPU server OOM / unavailable | Seamlessly route to Groq LLaMA-3-8B API fallback |
| **SMS Delivery** | Twilio rate limit / failed delivery | Queue failed SMS for retry; send Firebase Push as primary fallback |
| **Spatial DB** | PostGIS query latency | Cache district-to-pincode spatial mappings in Redis |
