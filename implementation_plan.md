# 🌡️ AI-Based Heatwave Prediction System — Project Synopsis & Architecture

> **Course**: Minor Project 1 (1 Credit) | **Team Goal**: Build a full-stack, AI-powered heatwave early warning system

---

## 📄 Synopsis

### Project Title
**HeatSentinel: An AI-Powered Heatwave Prediction and Early Warning System**

### Abstract
Climate change has intensified the frequency and severity of heatwaves globally, with India being among the most vulnerable nations. HeatSentinel is an intelligent, full-stack system that leverages machine learning to predict heatwaves up to **7 days in advance** using historical meteorological data sourced from reliable government databases (IMD, NASA POWER, NOAA). The system integrates a **fine-tuned small language model (SLM)** for natural language querying and summarization of predictions, a **real-time SMS/push notification alert system** for citizens in at-risk zones, and an **interactive geospatial dashboard** for visualization. The project bridges the gap between raw climate data and actionable public safety intelligence.

### Problem Statement
Heatwaves cause thousands of deaths annually in India. Existing alert systems are reactive and lack localized, hyperlocal prediction granularity. There is a need for a proactive, AI-driven system that can predict, communicate, and explain heatwave risks in plain language to both officials and citizens.

### Objectives
1. Collect and preprocess multi-year meteorological data from government APIs (IMD, NASA POWER).
2. Train a robust ML/DL model to predict heatwave events with ≥85% accuracy.
3. Integrate a fine-tuned SLM to enable natural language Q&A and report summarization.
4. Build a real-time alert system (SMS via Twilio/MSG91, push via Firebase FCM).
5. Deploy an interactive geospatial dashboard with risk maps, forecasts, and historical trends.
6. Expose all capabilities via a well-documented FastAPI backend.

### Scope
- Geographic Coverage: India (district/state level), with architecture scalable globally.
- Time Horizon: 1–7 day heatwave probability forecasts.
- Target Users: Government agencies, NDMA, citizens in high-risk zones.

---

## 🏗️ Refined System Architecture (Deep Technical Specification)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EXTERNAL DATA SOURCES                                  │
│  IMD APIs / CSVs │ NASA POWER (Solar/Wind) │ OpenMeteo ERA5 │ ISRO Bhuvan (NDVI) │ POSOCO    │
└───────────────────────────┬────────────────────────────────────────────────────────────┘
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

## 🔬 Architectural Layer Specifications

### 1. Data Ingestion & Preprocessing Layer
- **Sources**: IMD, NASA POWER, OpenMeteo ERA5, ISRO Bhuvan (NDVI), Census of India, POSOCO.
- **Formulas Computed On-The-Fly**:
  - *Heat Index (Rothfusz Regression)*: $HI = c_1 + c_2 T + c_3 RH + c_4 T \cdot RH + c_5 T^2 + \dots$
  - *WBGT Approximation*: $WBGT \approx 0.567 T + 0.393 e + 3.94$
  - *Rolling Statistics*: 3d, 7d, 14d rolling mean max temperature + consecutive hot days streak ($T \ge 40^\circ C$).

### 2. Dual-Model Stacking Ensemble & SHAP
- **XGBoost Sub-model**: Tabular classification (`is_heatwave`) & regression (`predicted_max_temp`).
- **Bi-LSTM Sub-model**: PyTorch sequence model over 14-day temporal windows capturing atmospheric momentum.
- **Meta-Learner**: Logistic regression stack blending probabilities into a single calibrated risk score.
- **SHAP Engine**: `shap.TreeExplainer` calculating exact feature attribution per prediction for frontend waterfall charts.

### 3. SLM Fine-Tuning & Multilingual Translation
- **Model**: Phi-3 Mini (3.8B) or Qwen2-1.5B fine-tuned via LoRA ($r=16, \alpha=32$).
- **Corpus**: IMD heatwave reports + synthetic prediction Q&A pairs + district geographic taxonomy.
- **IndicTrans2**: Translates generated alert messages into Hindi, Telugu, Tamil, Marathi, Bengali.

### 4. PostGIS Spatial Database & Celery Job Pipeline
- **Database**: PostgreSQL 15 + PostGIS extension.
- **Spatial Tables**: `districts` (MultiPolygon geometry) and `users` (Point geometry from pincode geocoding).
- **Spatial Query**: `ST_Contains(district.geom, user.location)` executed daily at 06:00 AM by Celery workers to fetch affected phone numbers.

### 5. React Frontend Architecture
- **Interactive Leaflet Map**: GeoJSON polygon overlay color-coded by real-time risk level.
- **Charts & Drawers**: Recharts 7-day forecast area charts, SHAP waterfall visualizer, SLM streaming chat widget, and OSM Overpass Cooling Center locator.

---

## 🧠 ML/DL Model Stack

### Primary Prediction Model
| Component | Choice | Why |
|-----------|--------|-----|
| **Tabular Baseline** | XGBoost | Handles missing data, fast training, interpretable |
| **Temporal Patterns** | LSTM / Bi-LSTM | Captures multi-day weather sequences |
| **Ensemble** | Stacking (XGB + LSTM) | Best of both worlds |
| **Explainability** | SHAP values | Shows which features drove prediction |

### Features (Input Variables)
- Max/Min/Mean Temperature (°C)
- Relative Humidity (%)
- Heat Index & Wet Bulb Globe Temperature (WBGT)
- Wind Speed & Direction
- Dew Point Temperature
- Solar Radiation (W/m²)
- NDVI (vegetation cover via Sentinel/MODIS)
- Urban Heat Island proxy (built-up area %)
- Consecutive dry days streak

### Target Labels
- `is_heatwave` (binary: IMD definition — temp ≥ 40°C and +4.5°C above normal for 2+ days)
- `heat_severity` (Mild / Moderate / Severe / Extreme)
- `predicted_max_temp` (regression)

---

## 🤖 SLM Integration Plan

### Model Choice
**Phi-3 Mini (3.8B)** or **Qwen2-1.5B** — both run on CPU/low-GPU

### Fine-Tuning Strategy
- **Method**: LoRA (Low-Rank Adaptation) via HuggingFace PEFT
- **Training Data**:
  - IMD heatwave reports (PDF → text)
  - Climate FAQ datasets
  - Synthetic Q&A pairs generated from prediction outputs
  - Geographic knowledge (district → state → climate zone mapping)

### Capabilities after fine-tuning
| Query Type | Example |
|------------|---------|
| Prediction Summary | *"Summarize the heatwave risk for Rajasthan this week"* |
| Citizen Advisory | *"What precautions should I take in Nagpur tomorrow?"* |
| Historical Q&A | *"How many heatwaves hit Odisha in 2023?"* |
| Technical Explanation | *"Why is the heat index higher than actual temperature?"* |

---

## 🔔 Alert & Notification System

### Architecture
1. **Trigger Logic**: If `heat_severity >= Severe` for a district → fire alert
2. **Channels**:
   - **SMS** → Twilio / MSG91 (Indian telco-friendly, cheaper)
   - **Push Notifications** → Firebase Cloud Messaging (FCM)
   - **Email Digest** → SendGrid (for officials/subscribers)
   - **Government Webhook** → POST to NDMA/state disaster mgmt APIs
3. **User Registration**: Citizens register phone number + pin code → stored in PostGIS DB
4. **Alert Frequency**: Max 1 alert/12 hrs per user to avoid spam
5. **Multilingual**: Alerts in Hindi + local language (use IndicTrans2 for translation)

---

## 📊 Data Sources (Government & Reliable)

| Source | Data | Access |
|--------|------|--------|
| **IMD (imdpune.gov.in)** | Historical temp, humidity, rainfall | Free API / CSV download |
| **NASA POWER** | Solar radiation, wind, temperature | Free REST API |
| **OpenMeteo** | Real-time + forecast (ERA5 reanalysis) | Free, no key needed |
| **NOAA GHCN** | Global historical station data | Free FTP |
| **Copernicus C3S** | ERA5 long-term reanalysis | Free (registration) |
| **Bhuvan (ISRO)** | India satellite imagery, NDVI | Free for Indian users |
| **Census of India** | Population density per district | Static CSV |

---

## 🌟 Feature List (Core + Elevators)

### Core Features
- [x] 7-day heatwave probability forecast (district-level)
- [x] Interactive choropleth map (India districts color-coded by risk)
- [x] Natural language chatbot for predictions & advisories
- [x] SMS + Push notification alert system with user registration
- [x] Historical heatwave trend analysis (charts)
- [x] SHAP-based explainability panel ("Why this prediction?")

### 🚀 Standout / Elevating Features

#### 1. **Vulnerability Index Overlay**
   - Combine heatwave risk with census data (elderly population %, outdoor workers %, slum density)
   - Produce a **Composite Heat Vulnerability Index (HVI)** per district
   - This is what NDMA actually uses — makes the project policy-relevant

#### 2. **Health Impact Forecasting**
   - Predict estimated heat-related illness/mortality risk using epidemiological models
   - "Severe heatwave in Vidarbha → ~200 heat stroke cases expected"

#### 3. **Cooling Center Locator**
   - Integrate Google Maps / OSM to show nearest government hospitals, water kiosks, shade zones
   - Triggered automatically when alert fires

#### 4. **Offline SMS-only Mode**
   - For rural users with no internet: receive heatwave alerts via plain SMS with actionable advice
   - No app install needed — massive real-world impact

#### 5. **Admin Dashboard for Officials**
   - Separate React dashboard for state officials: export PDF reports, view district-wise stats, trigger manual alerts
   - Role-based access (Admin / Viewer)

#### 6. **Model Drift Monitoring**
   - Track prediction accuracy daily using incoming actual temperature data
   - Alert devs if model accuracy drops below threshold (MLflow / Evidently AI)

#### 7. **Carbon + Energy Impact Tracker**
   - Track how heatwaves correlate with power grid stress (electricity demand spikes)
   - Source: POSOCO (Power System Operation Corporation) data

---

## 🛠️ Full Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Training** | Python, scikit-learn, XGBoost, PyTorch, HuggingFace |
| **SLM Fine-tuning** | HuggingFace PEFT (LoRA), Unsloth (faster training) |
| **Data Pipeline** | Pandas, NumPy, GeoPandas, Apache Airflow |
| **Backend API** | FastAPI, SQLAlchemy, Celery (async tasks), Redis |
| **Database** | PostgreSQL + PostGIS (geospatial queries) |
| **Frontend** | React, Recharts / Chart.js, Leaflet.js / Mapbox GL |
| **Notifications** | Twilio (SMS), Firebase FCM (push), SendGrid (email) |
| **Translation** | IndicTrans2 (AI4Bharat) |
| **MLOps** | MLflow (experiment tracking), Evidently AI (drift) |
| **Deployment** | Docker + docker-compose, HuggingFace Spaces (SLM) |
| **Auth** | JWT tokens (FastAPI), Firebase Auth (mobile) |

---

## 🗓️ 8-Week Team Sprint (4 Members)

### 👥 Team Role Assignment

| Member | Track | Owns |
|--------|-------|------|
| **Member A** | 🤖 ML & Data | Data pipeline, XGBoost, LSTM, SHAP, MLflow |
| **Member B** | 🧠 AI/LLM | SLM selection, LoRA fine-tuning, Groq/chat API, IndicTrans2 |
| **Member C** | ⚙️ Backend | FastAPI, PostgreSQL+PostGIS, Celery, Redis, Auth |
| **Member D** | 🎨 Frontend | React, Leaflet maps, Recharts, admin dashboard, UI polish |

> Everyone contributes to integration, testing, and the final report.

---

### 🗓️ Full 8-Week Sprint (All Features)

#### **Week 1 — Foundation & Data**
| Member | Tasks |
|--------|-------|
| A | Collect NASA POWER, IMD, OpenMeteo historical data for all Indian states. Clean & store as Parquet. |
| B | Research SLM options (Phi-3 Mini vs Qwen2-1.5B). Set up HuggingFace environment. Collect IMD PDF reports for fine-tuning corpus. |
| C | Set up PostgreSQL + PostGIS. Design DB schema (users, alerts, predictions, districts). Docker-compose skeleton. |
| D | Scaffold React app. Set up routing, design system (dark theme, fonts, color palette). Basic layout shells. |

#### **Week 2 — ML Model & Backend Core**
| Member | Tasks |
|--------|-------|
| A | Feature engineering (Heat Index, WBGT, NDVI, consecutive dry days, UHI proxy). Train XGBoost baseline. Evaluate (F1, ROC-AUC, precision-recall). |
| B | Prepare fine-tuning dataset — PDF scraping → text, synthetic Q&A generation from prediction outputs, geographic Q&A pairs. |
| C | Build core FastAPI endpoints: `/predict`, `/forecast/7day`, `/history`, `/districts`. SQLAlchemy models. |
| D | Build interactive Leaflet choropleth map with India district GeoJSON. Color-coded risk levels. Tooltip on hover. |

#### **Week 3 — LSTM + SLM Fine-tuning Begins**
| Member | Tasks |
|--------|-------|
| A | Add LSTM layer for temporal sequence modeling. Build XGBoost+LSTM stacking ensemble. SHAP value generation. |
| B | Start LoRA fine-tuning on Google Colab (free T4). Track training loss. First eval of climate Q&A quality. |
| C | Add `/register-alert`, `/chat`, `/admin` endpoints. Integrate Celery + Redis for async alert jobs. JWT auth. |
| D | Build 7-day forecast chart (Recharts), historical trend graphs, SHAP explainability panel in UI. |

#### **Week 4 — Alerts, Notifications & Heat Vulnerability Index**
| Member | Tasks |
|--------|-------|
| A | Build **Heat Vulnerability Index**: combine heatwave risk with census data (elderly %, slum density %, outdoor workers %). Validate against NDMA methodology. |
| B | LoRA fine-tuning complete. Evaluate on held-out Q&A set. Integrate into `/chat` endpoint via HuggingFace Inference or local server. |
| C | Integrate MSG91/Twilio SMS. Integrate Firebase FCM for push notifications. Alert logic: severity threshold → fire per-district. Max 1 alert/12hr per user. |
| D | Build user registration flow (phone + pincode). Build admin dashboard (district stats, manual alert trigger, export PDF report). |

#### **Week 5 — Multilingual, MLOps & Geospatial Features**
| Member | Tasks |
|--------|-------|
| A | Integrate MLflow for experiment tracking. Set up Evidently AI model drift monitoring (daily accuracy check vs actual IMD temps). |
| B | Integrate IndicTrans2 (AI4Bharat) for Hindi + regional language SMS translation. Test on 5 Indian languages. |
| C | PostGIS geospatial queries: find all users within a district polygon, radius-based alert targeting. Optimize query performance. |
| D | Cooling Center Locator using OpenStreetMap Overpass API (hospitals, water kiosks near user location). Wire into alert flow. |

#### **Week 6 — Health Impact, Carbon Tracker & Integration**
| Member | Tasks |
|--------|-------|
| A | Build **Health Impact Forecasting** model: correlate heat severity → expected heat-stroke incidence using historical IMD + health data. |
| B | Fine-tune SLM to answer health advisory queries. Add geographic district-level knowledge. Evaluate hallucination rate. |
| C | Integrate POSOCO data for **Carbon/Energy tracker** (heatwave ↔ grid stress correlation). Add `/energy-impact` endpoint. |
| D | Full UI integration of all new features — health panel, energy tracker, cooling center map layer, multilingual toggle. |

#### **Week 7 — End-to-End Testing & Polish**
| Member | Tasks |
|--------|-------|
| All | End-to-end integration testing. Fix all broken API connections. Load test FastAPI with Locust. |
| A | Write model evaluation report. Compare XGBoost vs LSTM vs ensemble. Final SHAP analysis. |
| B | Final SLM eval — accuracy, latency, hallucination. Compare with Groq baseline. |
| C | Security review — sanitize inputs, rate limiting, CORS. Production-ready Docker-compose. Deploy on Render/Railway. |
| D | UI polish — animations, loading skeletons, mobile responsiveness, PWA manifest. Dark mode. |

#### **Week 8 — Deployment, Demo & Report**
| Member | Tasks |
|--------|-------|
| A + B | Write ML + AI sections of the final report. Jupyter notebooks for EDA, model training, SLM fine-tuning. |
| C | Final deployment. Set up domain. Monitor server logs. Write API documentation (FastAPI auto-docs). |
| D | Record 5-min demo video walkthrough. Build project landing page / README. |
| All | Final report writing. Cross-review all sections. Submit. 🎉 |

---

## 🎯 What Makes This Stand Out (vs. Typical Minor Projects)

1. **Real Government Data** — not Kaggle toy datasets
2. **SLM Integration** — most minor projects skip LLM/AI entirely
3. **Multilingual SMS Alerts** — addresses Bharat (rural India), not just urban users
4. **Heat Vulnerability Index** — policy-relevant, matches NDMA's own framework
5. **Explainable AI** — SHAP values show WHY the model predicted a heatwave
6. **Full MLOps Loop** — model drift monitoring is grad-school level thinking
7. **PostGIS** — spatial database queries are rare in minor projects, impresses examiners

---

## 📋 Open Questions for Team Discussion

> [!IMPORTANT]
> **Decide before writing the synopsis**:
> 1. Will you build a mobile app (React Native) or just PWA (Progressive Web App)?
> 2. Do you have GPU access for SLM fine-tuning? (If not, use Phi-3 Mini on Google Colab)
> 3. Will this be a real deployed system or demo? (affects Twilio costs — they have free tier)
> 4. How many team members? (affects scope of features to commit to in synopsis)
> 5. Which state/region to focus on initially? (Rajasthan/Telangana are IMD heatwave hotspots)
