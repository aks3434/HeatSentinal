# 🎨 HeatSentinel — Responsive Frontend Dashboard Guide (Samriddhi's Module)

> **Assigned Developer**: Samriddhi  
> **Module**: Interactive Geospatial Dashboard & Citizen Web App (React + Leaflet + Recharts)  
> **Backend Status (Team Lead)**: ✅ 100% Live & Operational at `http://127.0.0.1:8000` (Swagger docs at `/docs`)

---

## 🛑 1. Git & Conflict-Prevention Rules (CRITICAL)

To prevent merge conflicts with the backend, ML models, and database:

1. **Your Workspace (Folder you own completely)**:
   - `frontend/` — Scaffold and build the entire React application inside this directory.

2. **Protected Files (DO NOT modify or overwrite)**:
   - ❌ `ml_engine/` — ML models (XGBoost, Bi-LSTM, Stacking Ensemble) are finalized.
   - ❌ `slm_engine/` — SLM inference and translation are finalized.
   - ❌ `backend/` — Managed by the Team Lead, Krishan (DB), and Pragya (Alerts).
   - ❌ `data/` — Contains model weights and datasets.

---

## 🎨 2. UI/UX Vision & Tech Stack

Build a sleek, modern, high-impact dashboard suitable for a top-tier project presentation:
- **Framework**: React 18+ (using Vite)
- **Styling**: TailwindCSS or modern CSS (sleek dark mode / glassmorphism accents, vibrant heat alert colors)
- **Mapping**: `react-leaflet` + `leaflet` (OpenStreetMap tiles with color-coded risk markers/polygons)
- **Charts**: `recharts` (for 7-day temperature trends & SHAP waterfall feature impact)
- **Icons**: `lucide-react`

---

## 🔗 3. Live Backend API Contract (Base URL: `http://127.0.0.1:8000`)

The Team Lead's backend has CORS enabled and is ready to accept requests from `http://localhost:5173`.

### Endpoint 1: Fetch Monitored Districts
- **GET** `/api/v1/predict/districts`
- **Response**:
```json
{
  "total": 19,
  "districts": [
    {
      "code": "RJ01",
      "name": "Jaipur",
      "state": "Rajasthan",
      "lat": 26.9124,
      "lon": 75.7873,
      "baseline_normal": 40.5
    }
  ]
}


Endpoint 3: SHAP Explainability Waterfall Data
GET /api/v1/explain/{district_name}
Response:
json
{
  "district": "Jaipur",
  "base_expected_risk": 0.18,
  "feature_attributions": [
    {
      "feature": "temp_max",
      "value": 44.5,
      "shap_value": 0.36,
      "impact": "increases_risk"
    },
    {
      "feature": "consecutive_hot_days",
      "value": 5.0,
      "shap_value": 0.35,
      "impact": "increases_risk"
    },
    {
      "feature": "wind_speed",
      "value": 12.0,
      "shap_value": -0.02,
      "impact": "decreases_risk"
    }
  ]
}
Endpoint 4: SLM Climate Chatbot
POST /api/v1/chat/query
Body: { "query": "What is the difference between dry-bulb temperature and Heat Index?" }
Response: { "query": "...", "response": "Heat Index measures..." }
Endpoint 5: Custom "What-If" Weather Simulator
POST /api/v1/predict/custom
Body:
json
{
  "district_code": "RJ01",
  "temp_max": 46.2,
  "humidity": 52.0,
  "wind_speed": 10.0,
  "solar_rad": 27.0,
  "consecutive_hot_days": 6
}
🧩 4. Core UI Components to Build
Navbar:

Title: HeatSentinel AI (with live status pulse ● Operational).
Language selector dropdown (English, हिंदी, मराठी, తెలుగు, தமிழ், বাংলা).
HeatRiskMap (Leaflet Map):

Centered on India ([20.5937, 78.9629], zoom level 5).
Markers for all 19 districts color-coded by severity:
🟢 Normal (< 40°C)
🟡 Moderate (40°C – 44.9°C)
🔴 Severe (45°C – 46.9°C)
🟣 Extreme (≥ 47°C)
Clicking a district updates the rest of the dashboard!
PredictionCard & HealthImpactWidget:

Shows Predicted Max Temp with thermometer graphic.
Stacking Ensemble probabilities breakdown gauge ($P_{\text{XGB}}$, $P_{\text{LSTM}}$, $P_{\text{Final}}$).
Hospital surge risk card: "243 Expected Heat Stroke Cases".
ShapExplainabilityChart:

Horizontal Recharts bar chart showing which features increased risk (red bars) or decreased risk (green bars).
SlmChatWidget:

Floating chat drawer where users can ask questions or view regional advisory bullet points.
WhatIfSimulator:

Interactive sliders for temperature (35°C–50°C), humidity (10%–90%), and hot day streaks.
Dynamically re-runs predictions as sliders move!
AlertSubscriptionModal:

A simple modal for citizens to enter their phone number and pincode to receive SMS alerts.
🚀 5. How Samriddhi Sets Up the Project
Run these commands in your terminal:

bash
# 1. Initialize Vite React app in the frontend directory
npm create vite@latest frontend -- --template react
# 2. Navigate to frontend
cd frontend
# 3. Install dependencies
npm install
npm install leaflet react-leaflet recharts lucide-react axios
# 4. Start the frontend dev server
npm run dev
Your React app will run on http://localhost:5173 and connect directly to the Team Lead's backend at http://127.0.0.1:8000!

