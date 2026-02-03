# Jal Drishti - Flood Simulation & Risk Assessment Dashboard

**Jal Drishti** (Water Vision) is a comprehensive Mission Control Dashboard designed for real-time flood monitoring, simulation, and risk assessment. It provides actionable intelligence through 3D visualization, predictive analytics, and simulation tools to assist in disaster management and planning.

## 🚀 Key Features

*   **Interactive 3D Map**: Powered by MapLibre GL JS, offering a detailed terrain view with village boundaries.
*   **Multi-Region Support**: specialized data and visualizations for:
    *   Meppadi (Wayanad, Kerala)
    *   Darbhanga (Bihar)
    *   Dhemaji (Assam)
*   **Flood Simulation Engine**:
    *   Interactive time-based flood progression.
    *   Adjustable rainfall intensity input (0-300mm).
    *   Visual representation of water levels over time.
*   **Risk Metrics & Analytics**:
    *   Real-time calculation of risk areas, exposed population, and soil saturation.
    *   Hydrological indicators: Elevation, Slope, and Hydro Index.
*   **Weather & Forecasting**:
    *   Live weather updates (Temperature, Wind, Humidity).
    *   Yearly and Short-term (7-day) precipitation forecasts.
    *   Visualized using interactive charts.
*   **Model Intelligence**:
    *   AI-driven insights on Terrain Stability and Potential Impact.
    *   Automated Simulation Report generation.

## 🛠️ Technology Stack

*   **Frontend**:
    *   **HTML5/CSS3**: Responsive glassmorphism UI design.
    *   **JavaScript (ES6+)**: Core logic and simulation engine.
    *   **MapLibre GL JS**: High-performance 3D mapping libraries.
    *   **Chart.js**: Data visualization for weather and forecasts.
*   **Backend/Data**:
    *   **Python**: Data processing and server (currently serving static files).
    *   **JSON/GeoJSON**: Spatial data formats for boundaries, buildings, and land use.

## 📂 Project Structure

```
jaldrishti/
├── dashboard/              # Main Frontend Application
│   ├── index.html          # Entry point
│   ├── css/                # Stylesheets
│   ├── js/                 # Application logic (simulation, map, UI)
│   ├── data/               # GeoJSON and JSON data files
│   ├── user-guide.html     # User documentation
│   └── methodology.html    # Scientific methodology explanation
├── src/                    # Backend & ML Core (In Development)
│   ├── core/               # Core application logic
│   ├── ml/                 # Machine learning models
│   ├── api/                # API endpoints
│   └── services/           # Auxiliary services
└── README.md               # Project Documentation
```

## ⚡ Setup & Usage

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Start the Local Server**:
    The application requires a local server to fetch data files correctly (to avoid CORS issues).
    ```bash
    python3 -m http.server 8000
    ```

3.  **Access the Dashboard**:
    Open your web browser and go to:
    `http://localhost:8000/dashboard`

## 📊 Data Sources

The dashboard utilizes high-resolution data for:
*   **Land Use/Land Cover**: JSON datasets for specific regions.
*   **Building Footprints**: Detailed building structures for impact analysis.
*   **Terrain Data**: AWS Terrarium for 3D elevation tiles.

## 🤝 Contribution

Contributions are welcome! Please feel free to verify the `src/` directory for ongoing backend developments and ML integration.
