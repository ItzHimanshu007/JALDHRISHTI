/**
 * Jal Drishti - Core Intelligence Engine (Phase 4 Professional)
 */

const UI_CONFIG = {
    // API Configuration
    apiBaseUrl: '',  // Relative path (same origin)

    colors: {
        extreme: '#7f1d1d',  // Darker red for extreme
        high: '#dc2626',
        medium: '#f59e0b',
        low: '#10b981',
        primary: '#0891b2', // Cyan-600
        secondary: '#06b6d4',
        rescue: '#22c55e',
        soilWet: '#3b82f6',
        soilDry: '#d97706'
    },

    // Photorealistic Map Configuration
    mapStyles: {
        // Satellite Imagery (Esri World Imagery - high quality)
        satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',

        // 3D Terrain (AWS Terrarium - elevation data)
        terrain: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',

        // Fallback raster style
        reliable: {
            "version": 8,
            "sources": {
                "satellite-base": {
                    "type": "raster",
                    "tiles": ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
                    "tileSize": 256,
                    "attribution": "Esri World Imagery"
                },
                "soil-saturation-source": {
                    "type": "geojson",
                    "data": { "type": "FeatureCollection", "features": [] }
                }
            },
            "layers": [
                {
                    "id": "background",
                    "type": "background",
                    "paint": { "background-color": "#020617" }
                },
                {
                    "id": "satellite-layer",
                    "type": "raster",
                    "source": "satellite-base",
                    "layout": { "visibility": "visible" },
                    "paint": { "raster-opacity": 1.0, "raster-saturation": -0.1 }
                }
            ]
        }
    },

    // Boundary styling for village outline
    boundaryStyle: {
        color: '#ff9500',
        glowColor: '#ffffff',
        width: 4,
        dashArray: [2, 1]
    }
};


let appState = {
    map: null,
    data: null,
    currentVillageId: 'wayanad_meppadi',
    currentTimeStep: '0h',
    rainfallAmount: 50, // mm - controls simulation intensity
    styleUrl: UI_CONFIG.mapStyles.reliable,
    charts: { rainfall: null, moisture: null, yearly: null, timeline: null },
    lastMoveUpdate: 0,
    moveDebounceTimeout: null,

    // Rescue Mode State
    rescueMode: false,
    rescuePath: null,

    // API Data Cache
    apiData: {
        boundary: null,
        infrastructure: null,
        population: null,
        floodSimulation: null
    },

    // Flood Animation State
    floodAnimationFrame: 0,
    floodOpacityDirection: 1
};

// ============================================
// API Integration & Data Fetching
// ============================================

/**
 * Fetch data from FastAPI backend with fallback to local data
 */
async function fetchFromAPI(endpoint, params = {}) {
    // Handle relative URLs correctly by providing a base if apiBaseUrl is empty
    const baseUrl = UI_CONFIG.apiBaseUrl || window.location.origin;
    const url = new URL(endpoint, baseUrl); // Construct URL relative to base

    Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));

    try {
        const response = await fetch(url, { timeout: 5000 });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        return await response.json();
    } catch (e) {
        console.warn(`API call failed: ${endpoint}`, e);
        return null;
    }
}

/**
 * Initialize dashboard from API or fallback to local data
 */
async function fetchDashboardData() {
    try {
        // Try API first
        const apiData = await fetchFromAPI('/api/init', { village_id: appState.currentVillageId });

        if (apiData && apiData.status === 'success') {
            // Store API data
            appState.apiData.boundary = apiData.boundary;
            appState.apiData.infrastructure = apiData.infrastructure;
            appState.apiData.population = apiData.population_heatmap;
            console.log('✓ Loaded data from API');
        }

        // Also load local data for village info
        const response = await fetch('data/dashboard_data.json');
        appState.data = await response.json();
        return true;
    } catch (e) {
        console.error("Data fetch failed, trying fallback", e);
        try {
            const response = await fetch('data/dashboard_data.json');
            appState.data = await response.json();
            return true;
        } catch (e2) {
            console.error("Critical: Could not load any data", e2);
            hideLoading();
            return false;
        }
    }
}

async function fetchLiveWeather(lat, lon) {
    if (!lat || !lon) return;

    // Update loading state
    const elTemp = document.getElementById('weatherTemp');
    const elDesc = document.getElementById('weatherDesc');
    if (elTemp) elTemp.style.opacity = '0.5';

    try {
        // Fetch comprehensive weather data including 7-day forecast
        const response = await fetch(
            `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
            `&current_weather=true` +
            `&hourly=relativehumidity_2m,precipitation,precipitation_probability` +
            `&daily=precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min,weathercode` +
            `&timezone=auto&forecast_days=7`
        );
        const data = await response.json();

        if (data.current_weather) {
            const current = data.current_weather;
            const temp = current.temperature;
            const wind = current.windspeed;
            const code = current.weathercode;

            // Get humidity (approximate from current hour)
            const currentHour = new Date().getHours();
            const humidity = data.hourly?.relativehumidity_2m?.[currentHour] || '--';

            // Update UI
            if (elTemp) {
                elTemp.textContent = `${temp.toFixed(1)}°C`;
                elTemp.style.opacity = '1';
            }

            const windEl = document.getElementById('weatherWind');
            const humEl = document.getElementById('weatherHum');
            if (windEl) windEl.textContent = wind;
            if (humEl) humEl.textContent = humidity;

            // Map WMO codes to icons/text
            const wmo = {
                0: { icon: '☀️', text: 'Clear Sky' },
                1: { icon: '🌤️', text: 'Mainly Clear' },
                2: { icon: '⛅', text: 'Partly Cloudy' },
                3: { icon: '☁️', text: 'Overcast' },
                45: { icon: '🌫️', text: 'Foggy' },
                48: { icon: '🌫️', text: 'Depositing Rime Fog' },
                51: { icon: '🌦️', text: 'Light Drizzle' },
                53: { icon: '🌦️', text: 'Moderate Drizzle' },
                55: { icon: '🌧️', text: 'Dense Drizzle' },
                61: { icon: '🌧️', text: 'Slight Rain' },
                63: { icon: '🌧️', text: 'Moderate Rain' },
                65: { icon: '⛈️', text: 'Heavy Rain' },
                71: { icon: '🌨️', text: 'Slight Snow' },
                73: { icon: '🌨️', text: 'Moderate Snow' },
                75: { icon: '❄️', text: 'Heavy Snow' },
                80: { icon: '🌧️', text: 'Rain Showers' },
                81: { icon: '🌧️', text: 'Moderate Showers' },
                82: { icon: '⛈️', text: 'Heavy Showers' },
                95: { icon: '⚡', text: 'Thunderstorm' },
                96: { icon: '⛈️', text: 'Thunderstorm & Hail' },
                99: { icon: '⛈️', text: 'Heavy Thunderstorm' },
            };

            const condition = wmo[code] || { icon: '❓', text: 'Unknown' };
            const iconEl = document.getElementById('weatherIcon');
            const descEl = document.getElementById('weatherDesc');
            if (iconEl) iconEl.textContent = condition.icon;
            if (descEl) descEl.textContent = condition.text;

            // Update Time
            const now = new Date();
            const timeEl = document.getElementById('weatherTime');
            if (timeEl) timeEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            // Store forecast data for short-term chart
            if (data.daily) {
                appState.liveWeatherForecast = {
                    dates: data.daily.time || [],
                    precipitation: data.daily.precipitation_sum || [],
                    probability: data.daily.precipitation_probability_max || [],
                    tempMax: data.daily.temperature_2m_max || [],
                    tempMin: data.daily.temperature_2m_min || [],
                    weatherCodes: data.daily.weathercode || [],
                    fetchedAt: new Date().toISOString()
                };
                
                // Update short-term chart with real data
                renderShortTermForecast();
            }

            console.log(`Live Weather Update [${appState.currentVillageId}]: ${temp}°C`);
        }

    } catch (e) {
        console.error("Weather fetch failed", e);
        if (elDesc) elDesc.textContent = "Data Unavailable";
    }
}

async function generateReport() {
    const btn = document.getElementById('btnGenerateReport');
    const originalText = btn.textContent;
    btn.textContent = 'Generating...';

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        // Header
        doc.setFillColor(15, 23, 42);
        doc.rect(0, 0, 210, 40, 'F');

        doc.setTextColor(34, 211, 238);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(22);
        doc.text("Jal Drishti", 20, 20);

        doc.setTextColor(255, 255, 255);
        doc.setFontSize(12);
        doc.text("FLOOD RISK INTELLIGENCE REPORT", 20, 30);

        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString()}`, 140, 30);

        // Capture Map (Low res for speed/pdf size)
        const mapCanvas = appState.map.getCanvas();
        const mapImg = mapCanvas.toDataURL("image/jpeg", 0.7);
        doc.addImage(mapImg, 'JPEG', 15, 50, 180, 100);

        // Village Info
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(14);
        doc.text(`Zone: ${document.getElementById('villageSelector').options[document.getElementById('villageSelector').selectedIndex].text}`, 20, 165);

        // Stats
        let y = 180;
        doc.setFontSize(10);
        const stats = [
            `Elevation: ${document.getElementById('valElev').textContent}`,
            `Slope: ${document.getElementById('valSlope').textContent}`,
            `Risk Area: ${document.getElementById('valArea').textContent}`,
            `Exposed Population: ${document.getElementById('totalPopRisk').textContent}`,
            `Flood Risk Level: ${document.getElementById('inspectRisk').textContent || 'N/A'}`
        ];

        stats.forEach(stat => {
            doc.text(`• ${stat}`, 25, y);
            y += 10;
        });

        doc.save(`JalDrishti_Report_${new Date().toISOString().slice(0, 10)}.pdf`);

    } catch (e) {
        console.error("Report generation failed", e);
        alert("Could not generate report. Browser security may block map capture.");
    } finally {
        btn.textContent = originalText;
    }
}



// ============================================
// 3D Map Engine (Enhanced with Fallbacks)
// ============================================

function init3DMap() {
    const startCoord = [76.135, 11.555]; // Wayanad

    appState.map = new maplibregl.Map({
        container: 'map',
        style: {
            "version": 8,
            "sources": {
                "satellite-source": {
                    "type": "raster",
                    "tiles": ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
                    "tileSize": 256,
                    "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community"
                },
                "soil-saturation-source": {
                    "type": "geojson",
                    "data": { "type": "FeatureCollection", "features": [] }
                }
            },
            "layers": [
                {
                    "id": "background",
                    "type": "background",
                    "paint": { "background-color": "#020617" }
                },
                {
                    "id": "satellite-layer",
                    "type": "raster",
                    "source": "satellite-source",
                    "paint": { "raster-opacity": 1.0, "raster-saturation": 0 }
                }
            ]
        },
        center: startCoord,
        zoom: 12,
        pitch: 60,
        bearing: -10,
        antialias: true,
        maxPitch: 85,
        fadeDuration: 0
    });

    // Expose map for verification/debugging
    window.map = appState.map;

    // Remove legacy fallback logic - we are enforcing satellite only
    appState.map.on('load', () => {
        // No-op for fallback
    });

    // Add Terrain for 3D display
    appState.map.on('styledata', () => {
        if (!appState.map.getSource('terrainSource')) {
            appState.map.addSource('terrainSource', {
                'type': 'raster-dem',
                'tiles': ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
                'encoding': 'terrarium',
                'tileSize': 256
            });
            const isTerrainActive = document.getElementById('btnLayerTerrain')?.classList.contains('active');
            appState.map.setTerrain({ 'source': 'terrainSource', 'exaggeration': isTerrainActive ? 1.5 : 0 });
        }
    });

    // Timeout safety
    const mapTimer = setTimeout(() => {
        console.warn("Map style slow to load, forcing UI display");
        hideLoading();
        // Don't call syncUI() here if map isn't ready, let the 'load' event handle it
        // Or if we MUST show something, ensure it doesn't crash on map layers
        if (appState.data) {
            // syncUI(); // Disabled here to prevent early map manipulation
            console.log("Data loaded, waiting for map to stabilize...");
        }

        // Fallback to OSM if Premium GL Style fails (likely source of empty tiles)
        if (!appState.map.isStyleLoaded() || appState.map.getCanvas().classList.contains('maplibregl-canvas-failing')) {
            console.warn("Falling back to robust OSM tiles due to potential GL style failure...");
            appState.map.setStyle(UI_CONFIG.mapStyles.reliable);
        }
    }, 4000);

    appState.map.on('load', () => {
        clearTimeout(mapTimer);
        initLayers();
        hideLoading();
        syncUI();

        // Render API-driven layers if available
        if (appState.apiData.boundary) {
            renderAPIBoundary();
        }
        if (appState.apiData.infrastructure) {
            renderAPIPOIs();
        }
        if (appState.apiData.population) {
            renderAPIPopulation();
        }

        console.log('✓ Map fully loaded with API layers');
    });

    appState.map.on('move', () => {
        const now = Date.now();
        if (now - appState.lastMoveUpdate > 100) { // Throttle updates
            appState.lastMoveUpdate = now;
            // Immediate sync if needed
        }
    });

    appState.map.on('error', (e) => {
        console.error("MapLibre Error:", e);
        // Fallback strategy
        if (!appState.map.loaded()) {
            hideLoading(); // Ensure UI is visible even if map fails
        }
    });
}

function initLayers() {
    // 1. Add 3D Building Extrusion - Removed (Requires vector source which is unavailable)

    // 2. Prepare Satellite Layer (Already added in style, but ensure visibility state)
    // 2. Prepare Satellite Layer (Ensure visibility)
    if (appState.map.getLayer('satellite-layer')) {
        appState.map.setLayoutProperty('satellite-layer', 'visibility', 'visible');
    }

    // 3. Remove Bhuvan Satellite (Unused/Unreliable)

    // 4. Pre-load ALL Risk Layers for smooth cross-fade
    const timeSteps = ['0h', '4h', '8h', '12h', '16h', '20h', '24h'];
    timeSteps.forEach(ts => {
        const layerId = `flood-risk-layer-${ts}`;
        if (!appState.map.getLayer(layerId)) {
            // Placeholder source, will be updated in updateMapVision
            appState.map.addSource(`flood-risk-source-${ts}`, { type: 'geojson', data: { "type": "FeatureCollection", "features": [] } });
            appState.map.addLayer({
                'id': layerId,
                'type': 'fill',
                'source': `flood-risk-source-${ts}`,
                'layout': { 'visibility': 'none' },
                'paint': {
                    'fill-color': [
                        'interpolate', ['linear'], ['get', 'value'],
                        0, 'rgba(34, 197, 94, 0.4)',
                        1, 'rgba(34, 197, 94, 0.6)',
                        2, 'rgba(234, 179, 8, 0.7)',
                        3, 'rgba(239, 68, 68, 0.8)',
                        4, 'rgba(127, 29, 29, 0.9)'
                    ],
                    'fill-opacity': 0,
                    'fill-antialias': true
                }
            });
        }

        appState.map.on('mouseenter', layerId, handleMouseEnter);
        appState.map.on('mouseleave', layerId, handleMouseLeave);
        appState.map.on('click', layerId, handleMapClick); // [NEW] Click listener for Deep Scan
    });

    // 5. Initialize Soil Layer (Hidden by default)
    if (!appState.map.getLayer('soil-saturation-layer')) {
        appState.map.addLayer({
            'id': 'soil-saturation-layer',
            'type': 'fill',
            'source': 'soil-saturation-source',
            'layout': { 'visibility': 'none' },
            'paint': {
                'fill-color': [
                    'interpolate', ['linear'], ['get', 'saturation'],
                    0, '#fef3c7',  // Dry - Light Yellow
                    0.4, '#60a5fa', // Moist - Blue
                    0.8, '#1e3a8a'  // Saturated - Dark Blue
                ],
                'fill-opacity': 0.6
            }
        });
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('hidden');
        // Hard remove after transition
        setTimeout(() => { overlay.style.display = 'none'; }, 600);
    }
}

function syncUI() {
    if (!appState.data || !appState.data.villages) return;
    let village = appState.data.villages[appState.currentVillageId];

    if (!village) {
        console.warn('Village not found:', appState.currentVillageId);
        return;
    }

    // Fetch Live Weather for this location (v2.4)
    fetchLiveWeather(village.info.coordinates.lat, village.info.coordinates.lon);

    // Header Info
    const elInfo = document.getElementById('villageInfo');
    if (elInfo && village.info) {
        elInfo.textContent = `${village.info.district}, ${village.info.state} | ${village.info.terrain_type.replace(/_/g, ' ')}`;
    }

    // High Density Data Grid Updates (REAL DATA)
    const stats = village.stats || {};
    const safeUpdate = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    safeUpdate('valElev', `${Math.round(stats.elevation_mean || 0)}m`);
    safeUpdate('valSlope', `${(stats.slope_max || 0).toFixed(1)}°`);
    safeUpdate('valRunoff', `${((stats.runoff_coefficient || 0.5) * 100).toFixed(1)}%`);

    const summary = village.forecast?.yearly?.yearly_summary || {};
    const riskArea = appState.data.is_live ? ((summary.peak_risk_score || 0) * 12.5) : ((summary.peak_risk_score || 0) * 3.2);
    safeUpdate('valArea', `${riskArea.toFixed(1)} km²`);

    // Enhanced Model Intelligence metrics
    const confidence = appState.data.model_metrics?.risk_scorer?.accuracy || 0.94;
    const terrainType = village.info.terrain_type || '';
    const isHilly = terrainType.includes('hilly') || terrainType.includes('ghats') || terrainType.includes('mountain');
    const isRiverine = terrainType.includes('plain') || terrainType.includes('flood') || terrainType.includes('river');
    
    // Terrain Analysis
    let terrainStatus = 'STABLE';
    let terrainColor = '#10b981';
    if (isHilly) {
        terrainStatus = 'DYNAMIC';
        terrainColor = '#f59e0b';
    }
    const stabilityEl = document.getElementById('intelStability');
    if (stabilityEl) {
        stabilityEl.textContent = terrainStatus;
        stabilityEl.style.color = terrainColor;
    }
    
    // Flood Type
    let floodType = 'RIVERINE';
    if (isHilly) floodType = 'FLASH FLOOD';
    else if (village.info.id === 'dhemaji') floodType = 'BRAHMAPUTRA';
    else if (village.info.id === 'darbhanga') floodType = 'KOSI BASIN';
    safeUpdate('intelFocus', floodType);
    
    // Impact Zone
    const impactZone = Math.round(riskArea * 1.4);
    safeUpdate('intelImpact', `${impactZone} km²`);
    
    // Confidence Score
    const confidencePercent = Math.round(confidence * 100);
    const confidenceEl = document.getElementById('intelConfidence');
    if (confidenceEl) {
        confidenceEl.textContent = `${confidencePercent}%`;
        confidenceEl.style.color = confidencePercent > 90 ? '#10b981' : (confidencePercent > 80 ? '#f59e0b' : '#ef4444');
    }
    
    // Model Accuracy Bar
    const accuracyValueEl = document.getElementById('modelAccuracyValue');
    const accuracyFillEl = document.getElementById('accuracyFill');
    if (accuracyValueEl) {
        accuracyValueEl.textContent = `${confidencePercent}%`;
    }
    if (accuracyFillEl) {
        setTimeout(() => {
            accuracyFillEl.style.width = `${confidencePercent}%`;
        }, 300);
    }

    // ML Accuracy Panel removed as requested. AccuracyBadge logic moved to separate panel if needed, but removed from header.

    // Stats & Population
    const popRisk = appState.data.is_live ? (village.info.population * 0.05) : (village.info.population * (summary.flood_probability || 0));
    safeUpdate('totalPopRisk', Math.round(popRisk).toLocaleString());

    // Live Weather & Soil Data
    if (stats.live_data) {
        const weather = stats.live_data.weather;
        const soil = stats.live_data.soil;

        if (weather && weather.current) {
            // Temperature removed from telemetry as requested
        }

        if (soil) {
            safeUpdate('valSoil', `${(soil.soil_moisture_m3_m3 * 100).toFixed(1)}%`);
        } else {
            safeUpdate('valSoil', '25.0%');
        }
    } else {
        safeUpdate('valSoil', '--');
    }

    // Live Status removed from header as requested

    // Charts
    renderEnhancedCharts(village);
    renderShortTermForecast();

    // Update Weekly/Monthly Badges based on actual forecast data
    const forecast = appState.liveWeatherForecast;
    let weeklyStatus = 'LOADING';
    
    if (forecast && forecast.precipitation) {
        const weeklyPrecip = forecast.precipitation.reduce((a, b) => a + b, 0);
        const maxDayPrecip = Math.max(...forecast.precipitation);
        const highProbDays = forecast.probability.filter(p => p > 60).length;
        
        if (maxDayPrecip > 50 || weeklyPrecip > 150) {
            weeklyStatus = 'WARNING';
        } else if (maxDayPrecip > 20 || highProbDays >= 3) {
            weeklyStatus = 'WATCH';
        } else if (weeklyPrecip > 30 || highProbDays >= 1) {
            weeklyStatus = 'MONITOR';
        } else {
            weeklyStatus = 'STABLE';
        }
    }
    
    const weeklyEl = document.getElementById('valWeekly');
    if (weeklyEl) {
        weeklyEl.textContent = weeklyStatus;
        // Color based on status
        if (weeklyStatus === 'WARNING') weeklyEl.style.color = '#ef4444';
        else if (weeklyStatus === 'WATCH') weeklyEl.style.color = '#f97316';
        else if (weeklyStatus === 'MONITOR') weeklyEl.style.color = '#eab308';
        else weeklyEl.style.color = '#22c55e';
    }

    const monthlyProb = (village.forecast?.yearly?.yearly_summary?.flood_probability * 100).toFixed(0);
    safeUpdate('valMonthly', `${monthlyProb} %`);

    // Update Chart Title Header with more info
    const chartTitle = document.querySelector('#yearlyChart').closest('.glass-panel').querySelector('.panel-header');
    if (chartTitle) {
        const yearlySummary = village.forecast?.yearly?.yearly_summary || {};
        const totalRainfall = Math.round(yearlySummary.expected_rainfall_mm || 0);
        const peakMonth = yearlySummary.peak_risk_month || 'July';
        const highRiskDays = yearlySummary.total_high_risk_days || 0;
        chartTitle.innerHTML = `
            <span style="display:flex; align-items:center; gap:6px;">
                <span class="layer-icon">📊</span> 
                ${village.info.name} - 2026 Forecast
            </span>
            <span style="font-size:0.65rem; color:var(--text-muted); font-weight:400;">
                Total: ${totalRainfall.toLocaleString()}mm | Peak: ${peakMonth} | ⚠️${highRiskDays} risk days
            </span>
        `;
    }

    updateMapVision(village);
    generateSoilGrid(village); // [NEW] Generate soil data
}
function renderEnhancedCharts(village) {
    const months = village.forecast?.yearly?.monthly_forecast || [];
    if (!months.length) {
        console.warn('No monthly forecast data for village:', village.info?.id);
        return;
    }

    const cvsYearly = document.getElementById('yearlyChart');
    if (cvsYearly) {
        const ctxYearly = cvsYearly.getContext('2d');
        if (appState.charts.yearly) appState.charts.yearly.destroy();

        // Calculate yearly stats
        const yearlySummary = village.forecast?.yearly?.yearly_summary || {};
        const totalRainfall = months.reduce((sum, m) => sum + m.expected_rainfall_mm, 0);
        const peakMonth = yearlySummary.peak_risk_month || 'July';
        const highRiskMonths = yearlySummary.high_risk_months || [];

        appState.charts.yearly = new Chart(ctxYearly, {
            type: 'bar',
            data: {
                labels: months.map(m => m.month_name.substring(0, 3)),
                datasets: [
                    {
                        label: 'Expected Rainfall (mm)',
                        data: months.map(m => m.expected_rainfall_mm),
                        backgroundColor: months.map(m => {
                            // Color based on risk level
                            const risk = m.risk_level || 'low';
                            if (risk === 'extreme') return 'rgba(239, 68, 68, 0.8)';
                            if (risk === 'high') return 'rgba(249, 115, 22, 0.8)';
                            if (risk === 'medium') return 'rgba(234, 179, 8, 0.7)';
                            return 'rgba(34, 197, 94, 0.6)';
                        }),
                        borderColor: months.map(m => {
                            const risk = m.risk_level || 'low';
                            if (risk === 'extreme') return '#ef4444';
                            if (risk === 'high') return '#f97316';
                            if (risk === 'medium') return '#eab308';
                            return '#22c55e';
                        }),
                        borderWidth: 2,
                        borderRadius: 4,
                        order: 2
                    },
                    {
                        label: 'Flood Probability %',
                        data: months.map(m => (m.flood_probability || 0) * 100),
                        type: 'line',
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: months.map(m => {
                            const prob = m.flood_probability || 0;
                            if (prob > 0.7) return '#ef4444';
                            if (prob > 0.4) return '#f97316';
                            return '#06b6d4';
                        }),
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'y1',
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true, 
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: { 
                    legend: { 
                        display: true,
                        position: 'top',
                        labels: {
                            color: 'rgba(255,255,255,0.7)',
                            font: { size: 9 },
                            boxWidth: 12,
                            padding: 8
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#fff',
                        bodyColor: 'rgba(255,255,255,0.8)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            title: (items) => {
                                const month = months[items[0].dataIndex];
                                const season = month?.season ? month.season.replace('_', ' ').toUpperCase() : '';
                                return `${month?.month_name || ''} ${season ? '- ' + season : ''}`;
                            },
                            afterBody: (items) => {
                                const month = months[items[0].dataIndex];
                                if (!month) return [];
                                const lines = [];
                                lines.push(`Risk Level: ${(month.risk_level || 'low').toUpperCase()}`);
                                lines.push(`High Risk Days: ${month.high_risk_days || 0}`);
                                if (month.alerts && month.alerts.length > 0) {
                                    lines.push(`⚠️ ${month.alerts[0].message}`);
                                }
                                return lines;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        display: true,
                        beginAtZero: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Rainfall (mm)',
                            color: 'rgba(255,255,255,0.5)',
                            font: { size: 9 }
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 9 } }
                    },
                    y1: {
                        display: true,
                        beginAtZero: true,
                        max: 100,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Flood Prob %',
                            color: 'rgba(6, 182, 212, 0.7)',
                            font: { size: 9 }
                        },
                        grid: { display: false },
                        ticks: { 
                            color: 'rgba(6, 182, 212, 0.7)', 
                            font: { size: 9 },
                            callback: (val) => val + '%'
                        }
                    },
                    x: {
                        display: true,
                        grid: { display: false },
                        ticks: { 
                            color: 'rgba(255,255,255,0.5)', 
                            font: { size: 9 },
                            callback: function(val, index) {
                                const month = months[index];
                                // Highlight peak risk month
                                if (month.month_name === peakMonth) {
                                    return '⚠️' + month.month_name.substring(0, 3);
                                }
                                return month.month_name.substring(0, 3);
                            }
                        }
                    }
                }
            }
        });
    }
}

function renderShortTermForecast() {
    const cvs = document.getElementById('shortTermChart');
    if (!cvs) return;

    const ctx = cvs.getContext('2d');
    if (appState.charts.shortTerm) appState.charts.shortTerm.destroy();

    // Use live weather forecast data if available
    const forecast = appState.liveWeatherForecast;
    
    let labels = [];
    let precipData = [];
    let probData = [];
    let tempMaxData = [];
    let tempMinData = [];
    
    if (forecast && forecast.dates && forecast.dates.length > 0) {
        // Use real API data
        labels = forecast.dates.map(d => {
            const date = new Date(d);
            return date.toLocaleDateString('en-US', { weekday: 'short' });
        });
        precipData = forecast.precipitation.map(p => p || 0);
        probData = forecast.probability.map(p => p || 0);
        tempMaxData = forecast.tempMax || [];
        tempMinData = forecast.tempMin || [];
    } else {
        // Fallback - show placeholder until data loads
        labels = ['Today', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
        precipData = [0, 0, 0, 0, 0, 0, 0];
        probData = [0, 0, 0, 0, 0, 0, 0];
    }
    
    const total = precipData.reduce((a, b) => a + b, 0).toFixed(1);
    const maxPrecip = Math.max(...precipData);
    const highRiskDays = precipData.filter(p => p > 20).length;

    appState.charts.shortTerm = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Rainfall (mm)',
                    data: precipData,
                    backgroundColor: precipData.map(p => {
                        if (p > 50) return 'rgba(239, 68, 68, 0.8)';
                        if (p > 20) return 'rgba(249, 115, 22, 0.8)';
                        if (p > 5) return 'rgba(6, 182, 212, 0.7)';
                        return 'rgba(34, 197, 94, 0.6)';
                    }),
                    borderColor: precipData.map(p => {
                        if (p > 50) return '#ef4444';
                        if (p > 20) return '#f97316';
                        if (p > 5) return '#06b6d4';
                        return '#22c55e';
                    }),
                    borderWidth: 2,
                    borderRadius: 4,
                    order: 2
                },
                {
                    label: 'Rain Probability %',
                    data: probData,
                    type: 'line',
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: probData.map(p => {
                        if (p > 70) return '#ef4444';
                        if (p > 40) return '#f97316';
                        return '#a855f7';
                    }),
                    tension: 0.3,
                    fill: true,
                    yAxisID: 'y1',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true, 
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: { 
                legend: { 
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(255,255,255,0.7)',
                        font: { size: 8 },
                        boxWidth: 10,
                        padding: 6
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#fff',
                    bodyColor: 'rgba(255,255,255,0.8)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 8,
                    callbacks: {
                        title: (items) => {
                            if (forecast && forecast.dates) {
                                const date = new Date(forecast.dates[items[0].dataIndex]);
                                return date.toLocaleDateString('en-US', { 
                                    weekday: 'long', 
                                    month: 'short', 
                                    day: 'numeric' 
                                });
                            }
                            return items[0].label;
                        },
                        afterBody: (items) => {
                            const idx = items[0].dataIndex;
                            const lines = [];
                            if (tempMaxData[idx] !== undefined) {
                                lines.push(`Temperature: ${tempMinData[idx]}°C - ${tempMaxData[idx]}°C`);
                            }
                            const precip = precipData[idx];
                            if (precip > 50) lines.push('⚠️ Heavy rainfall expected');
                            else if (precip > 20) lines.push('🌧️ Moderate rainfall');
                            else if (precip > 5) lines.push('🌦️ Light rainfall');
                            else lines.push('☀️ Mostly dry');
                            return lines;
                        }
                    }
                }
            },
            scales: {
                y: { 
                    beginAtZero: true, 
                    position: 'left',
                    title: {
                        display: true,
                        text: 'mm',
                        color: 'rgba(255,255,255,0.4)',
                        font: { size: 8 }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 8 } } 
                },
                y1: {
                    beginAtZero: true,
                    max: 100,
                    position: 'right',
                    title: {
                        display: true,
                        text: '%',
                        color: 'rgba(168, 85, 247, 0.7)',
                        font: { size: 8 }
                    },
                    grid: { display: false },
                    ticks: { 
                        color: 'rgba(168, 85, 247, 0.7)', 
                        font: { size: 8 },
                        callback: (val) => val + '%'
                    }
                },
                x: { 
                    grid: { display: false }, 
                    ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 8 } } 
                }
            }
        }
    });

    // Update panel header with forecast summary
    const panelHeader = cvs.closest('.glass-panel')?.querySelector('.panel-header');
    if (panelHeader && forecast && forecast.fetchedAt) {
        const fetchTime = new Date(forecast.fetchedAt);
        panelHeader.innerHTML = `
            <span>Short-term Rainfall (7-Day)</span>
            <span style="font-size:0.6rem; color:var(--text-muted); font-weight:400;">
                Total: ${total}mm | ${highRiskDays > 0 ? '⚠️' + highRiskDays + ' rainy days' : '☀️ Mostly dry'}
            </span>
        `;
    }

    const summaryEl = document.getElementById('forecastSummary');
    if (summaryEl) {
        if (maxPrecip > 50) {
            summaryEl.innerHTML = `<span style="color:#ef4444">⚠️ Heavy rain expected: ${total}mm total</span>`;
        } else if (maxPrecip > 20) {
            summaryEl.innerHTML = `<span style="color:#f97316">🌧️ Moderate rain: ${total}mm total</span>`;
        } else if (total > 0) {
            summaryEl.textContent = `Expected: ${total}mm over 7 days`;
        } else {
            summaryEl.textContent = `Loading forecast...`;
        }
    }
}


function updateMapVision(village) {
    if (!appState.map) return;

    let coords = [76.135, 11.555]; // Default Wayanad
    if (village && village.info && village.info.coordinates) {
        coords = [village.info.coordinates.lon, village.info.coordinates.lat];
    }

    // Only FLY to location if village changed or explicitly requested
    if (appState._lastFlownVillageId !== appState.currentVillageId) {
        if (appState.moveDebounceTimeout) clearTimeout(appState.moveDebounceTimeout);
        appState.moveDebounceTimeout = setTimeout(() => {
            appState.map.flyTo({
                center: coords,
                zoom: 13.5,
                pitch: 65,
                essential: true
            });
            appState._lastFlownVillageId = appState.currentVillageId;
        }, 50);
    }

    if (!village) return;

    // Add Village Boundary Outline
    addVillageBoundary(appState.currentVillageId);

    const isRiskVisible = document.getElementById('btnLayerRisk')?.classList.contains('active');

    // Optimized Map Vision Update
    const timeSteps = ['0h', '4h', '8h', '12h', '16h', '20h', '24h'];
    const currentStep = appState.currentTimeStep;

    // Only update the active timestep grid for performance, 
    // others can be updated on-demand or background
    timeSteps.forEach(ts => {
        const layerId = `flood-risk-layer-${ts}`;
        const sourceId = `flood-risk-source-${ts}`;

        if (ts === currentStep && isRiskVisible) {
            let timeFactor = 0;
            if (ts === '4h') timeFactor = 0.16;
            if (ts === '8h') timeFactor = 0.33;
            if (ts === '12h') timeFactor = 0.5;
            if (ts === '16h') timeFactor = 0.66;
            if (ts === '20h') timeFactor = 0.83;
            if (ts === '24h') timeFactor = 1.0;

            const source = appState.map.getSource(sourceId);
            if (source) {
                // Try to load API data if available
                if (appState.apiData.floodSimulation) {
                    // Filter simulation features for current timestep
                    const tsData = {
                        type: 'FeatureCollection',
                        features: appState.apiData.floodSimulation.features.filter(
                            f => f.properties.timestep === ts || f.properties.timestep === `t${timeSteps.indexOf(ts) + 1}`
                        )
                    };

                    if (source._lastCacheKey !== `api_${appState.currentVillageId}_${ts}_${appState.rainfallAmount}`) {
                        source.setData(tsData);
                        source._lastCacheKey = `api_${appState.currentVillageId}_${ts}_${appState.rainfallAmount}`;
                    }
                } else {
                    // Try to load real data if available in village object
                    const geojsonKey = `geojson_${ts}`;
                    const dataUrl = village[geojsonKey] ? `${village[geojsonKey]}` : null;

                    if (dataUrl) {
                        // Caching: Avoid redundant setData calls for the same URL
                        if (source._lastUrl !== dataUrl) {
                            source.setData(dataUrl);
                            source._lastUrl = dataUrl;
                        }
                    } else {
                        // Fallback to synthetic
                        const intensity = (appState.rainfallAmount / 300) * timeFactor;
                        // Cache synthetic data by intensity bucket to prevent constant regeneration
                        const intensityBucket = Math.round(intensity * 10) / 10;
                        const cacheKey = `${appState.currentVillageId}_${ts}_${intensityBucket}`;

                        if (source._lastCacheKey !== cacheKey) {
                            const floodData = generateFloodGrid(coords, intensity, appState.currentVillageId);
                            console.log(`[DEBUG] Generated ${floodData.features.length} flood tiles for ${cacheKey}`);
                            if (floodData.features.length > 0) {
                                console.log('[DEBUG] First tile props:', floodData.features[0].properties);
                            }
                            source.setData(floodData);
                            source._lastCacheKey = cacheKey;
                        }
                    }
                }
            }

            if (appState.map.getLayer(layerId)) {
                appState.map.setLayoutProperty(layerId, 'visibility', 'visible');
                const baseOpacity = 0.65; // Increased base visibility
                const rainfallFactor = Math.min(1.2, Math.max(0.5, appState.rainfallAmount / 150));
                appState.map.setPaintProperty(layerId, 'fill-opacity', Math.min(0.9, baseOpacity * rainfallFactor));
            }

            if (ts === '24h') checkRiskForVoiceAlert(village);
        } else {
            // Hide inactive layers immediately
            if (appState.map.getLayer(layerId)) {
                appState.map.setLayoutProperty(layerId, 'visibility', 'none');
                appState.map.setPaintProperty(layerId, 'fill-opacity', 0);
            }
        }
    });

    // Cleanup legacy layers
    if (appState.map.getLayer('flow-vectors-layer')) {
        appState.map.removeLayer('flow-vectors-layer');
        appState.map.removeSource('flow-vectors-source');
    }

    updateAnalyticsLayers(village);

    // Terrain Base Layer for Deep Scan
    if (village.terrain_geojson) {
        const terrainUrl = `${village.terrain_geojson}`;
        if (!appState.map.getSource('terrain-base-source')) {
            appState.map.addSource('terrain-base-source', { type: 'geojson', data: terrainUrl });
            appState.map.addLayer({
                'id': 'terrain-base-layer',
                'type': 'circle',
                'source': 'terrain-base-source',
                'paint': {
                    'circle-radius': 5,
                    'circle-opacity': 0,
                    'circle-color': '#fff'
                }
            });
        } else {
            const source = appState.map.getSource('terrain-base-source');
            if (source && source._lastUrl !== terrainUrl) {
                source.setData(terrainUrl);
                source._lastUrl = terrainUrl;
            }
        }
    }
}

function addVillageBoundary(villageId) {
    if (!appState.map) return;

    // Favor API data
    if (appState.apiData.boundary) {
        renderAPIBoundary();
        return;
    }

    const village = appState.data?.villages[villageId];
    const boundaryPath = village?.boundary_geojson ? `${village.boundary_geojson}` : `data/raw/boundaries/${villageId}_boundary.geojson`;

    // Attempt to load boundary GeoJSON
    if (!appState.map.getSource('village-boundary-source')) {
        appState.map.addSource('village-boundary-source', {
            type: 'geojson',
            data: boundaryPath
        });

        appState.map.addLayer({
            'id': 'village-boundary-layer',
            'type': 'line',
            'source': 'village-boundary-source',
            'paint': {
                'line-color': '#ffcc00',
                'line-width': 3,
                'line-opacity': 0.8,
                'line-dasharray': [2, 1]
            }
        });

        appState.map.addLayer({
            'id': 'village-boundary-glow',
            'type': 'line',
            'source': 'village-boundary-source',
            'paint': {
                'line-color': '#ffffff',
                'line-width': 10,
                'line-opacity': 0.2,
                'line-blur': 6
            }
        });
    } else {
        const source = appState.map.getSource('village-boundary-source');
        if (source && source._lastUrl !== boundaryPath) {
            source.setData(boundaryPath);
            source._lastUrl = boundaryPath;
        }
    }
}

function updateAnalyticsLayers(village) {
    if (!appState.map || !village) return;

    const isPopVisible = document.getElementById('btnLayerPop')?.classList.contains('active') ? 'visible' : 'none';

    // Favor API data
    if (appState.apiData.population) {
        renderAPIPopulation();
        if (appState.map.getLayer('api-population-layer')) {
            appState.map.setLayoutProperty('api-population-layer', 'visibility', isPopVisible);
        }
        return;
    }

    if (village.population_geojson) {
        const popUrl = `../${village.population_geojson}`;
        if (!appState.map.getSource('population-source')) {
            appState.map.addSource('population-source', { type: 'geojson', data: popUrl });

            appState.map.addLayer({
                'id': 'population-layer',
                'type': 'heatmap',
                'source': 'population-source',
                'layout': { 'visibility': isPopVisible },
                'paint': {
                    'heatmap-weight': ['interpolate', ['linear'], ['get', 'weight'], 0, 0, 1, 1],
                    'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 15, 3],
                    'heatmap-color': [
                        'interpolate', ['linear'], ['heatmap-density'],
                        0, 'rgba(0, 0, 255, 0)',
                        0.2, 'rgba(65, 105, 225, 0.3)',
                        0.4, 'rgba(0, 255, 255, 0.5)',
                        0.6, 'rgba(0, 255, 0, 0.6)',
                        0.8, 'rgba(255, 255, 0, 0.7)',
                        1, 'rgba(255, 0, 0, 0.8)'
                    ],
                    'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 2, 10, 15, 15, 35],
                    'heatmap-opacity': 0.7
                }
            });
        } else {
            const source = appState.map.getSource('population-source');
            if (source && source._lastUrl !== popUrl) {
                source.setData(popUrl);
                source._lastUrl = popUrl;
            }
        }
        appState.map.setLayoutProperty('population-layer', 'visibility', isPopVisible);
    }
}

// Function removed (Feature disabled as per user request)
function animateFlowVectors() { }

// Function removed (Feature disabled as per user request)
async function integrateWeatherRadar() {
    console.log("Weather Radar disabled by user configuration.");
}

// [NEW] Soil Saturation Generator
function generateSoilGrid(village) {
    if (!appState.map || !village) return;

    // Check if layer is active
    const isSoilVisible = document.getElementById('btnLayerSoil')?.classList.contains('active');
    if (!isSoilVisible) {
        if (appState.map.getLayer('soil-saturation-layer')) {
            appState.map.setLayoutProperty('soil-saturation-layer', 'visibility', 'none');
        }
        return;
    }

    const config = SIMULATION_CONFIG[appState.currentVillageId] || SIMULATION_CONFIG.wayanad_meppadi;
    const bbox = config.bbox;
    const gridSize = 25; // Coarser grid for soil

    // Use stored climate data
    const baseSoil = VILLAGE_CLIMATE[appState.currentVillageId]?.baseSoil || 0.35;

    const features = [];
    const stepLon = (bbox[2] - bbox[0]) / gridSize;
    const stepLat = (bbox[3] - bbox[1]) / gridSize;

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const lon = bbox[0] + (x * stepLon);
            const lat = bbox[1] + (y * stepLat);

            // Generate varied soil moisture
            // More moisture near water bodies (low terrain) or valleys
            const noise = Math.sin(x * 0.5) * Math.cos(y * 0.5);
            let saturation = baseSoil + (noise * 0.2);

            // Increase saturation if raining
            saturation += (appState.rainfallAmount / 500);

            if (saturation > 1) saturation = 1;
            if (saturation < 0) saturation = 0;

            features.push({
                type: 'Feature',
                properties: { saturation: saturation },
                geometry: {
                    type: 'Polygon',
                    coordinates: [[
                        [lon, lat],
                        [lon + stepLon, lat],
                        [lon + stepLon, lat + stepLat],
                        [lon, lat + stepLat],
                        [lon, lat]
                    ]]
                }
            });
        }
    }

    const source = appState.map.getSource('soil-saturation-source');
    if (source) {
        source.setData({ type: 'FeatureCollection', features: features });
        if (appState.map.getLayer('soil-saturation-layer')) {
            appState.map.setLayoutProperty('soil-saturation-layer', 'visibility', 'visible');
        }
    }
}

// ============================================
// Interactions (Simplified)
// ============================================
let currentPopup = null;

// Deep Scan removed

function handleMouseEnter(e) {
    appState.map.getCanvas().style.cursor = 'crosshair';
}

function handleMapClick(e) {
    // Check if deep scan is possible (simulation active)
    handleDeepScan(null, e.lngLat);
}

function handleMouseLeave() {
    appState.map.getCanvas().style.cursor = '';
    if (appState.map.getLayer('scan-highlight-layer')) {
        appState.map.setLayoutProperty('scan-highlight-layer', 'visibility', 'none');
    }
}

function closeInspector() {
    document.getElementById('cellInspector').style.display = 'none';
    if (currentPopup) currentPopup.remove();
}

// ============================================
// Initialization
// ============================================

// ============================================
// Initialization & HUD Wiring
// ============================================

function bindEvents() {
    // 1. Selector
    document.getElementById('villageSelector').addEventListener('change', async (e) => {
        if (e.target.value === 'custom') return;
        appState.currentVillageId = e.target.value;

        // Re-initialize for new village
        await fetchDashboardData();

        syncUI();
        updateSyntheticData(); // Update temp/soil for new village

        // Ensure map updates vision for new coordinates
        const village = appState.data.villages[appState.currentVillageId];
        updateMapVision(village);
    });


    // Report
    // Report
    document.getElementById('btnGenerateReport').addEventListener('click', () => {
        const village = appState.data.villages[appState.currentVillageId];
        generateSimulationReport(village);
    });

    // Opacity Slider
    // Opacity Slider removed

    // 2. Timeline Player Logic
    const playBtn = document.getElementById('masterPlayBtn');
    const timeDisplay = document.getElementById('timeDisplay');
    const timelineFill = document.getElementById('timelineFill');
    const ticks = document.querySelectorAll('.tick');

    playBtn.addEventListener('click', () => {
        if (appState.animationId) {
            // STOP
            clearInterval(appState.animationId);
            appState.animationId = null;
            playBtn.textContent = '▶';
            playBtn.style.background = 'var(--accent-primary)';
            playBtn.style.color = 'white';
        } else {
            // PLAY
            playBtn.textContent = '⏸';
            playBtn.style.background = 'var(--bg-elevated)';
            playBtn.style.color = 'var(--accent-secondary)';

            const labels = ['0h', '4h', '8h', '12h', '16h', '20h', '24h'];
            appState.animationId = setInterval(() => {
                let currentIdx = labels.indexOf(appState.currentTimeStep);
                let nextIdx = (currentIdx + 1) % labels.length;
                appState.currentTimeStep = labels[nextIdx];

                // Update UI state w/o triggering redundant events
                updateTimeUI(nextIdx);
                const village = appState.data.villages[appState.currentVillageId];
                updateMapVision(village);

                // Rescue path removed

                // Generate report at end of simulation cycle (24h)
                // Auto-report disabled as per user request
                // if (nextIdx === 3) { generateSimulationReport(village); }
            }, 2500);
        }
    });

    // Click on timeline ticks
    ticks.forEach(tick => {
        tick.addEventListener('click', () => {
            const idx = parseInt(tick.dataset.t);
            const labels = ['0h', '4h', '8h', '12h', '16h', '20h', '24h'];
            appState.currentTimeStep = labels[idx];
            updateTimeUI(idx);
            updateMapVision(appState.data.villages[appState.currentVillageId]);
        });
    });

    // 3. Layer Commanders removed

    // 4. Rainfall Slider
    const rainfallSlider = document.getElementById('rainfallSlider');
    const rainfallValue = document.getElementById('rainfallValue');
    if (rainfallSlider) {
        rainfallSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            if (appState.rainfallAmount === val) return;

            appState.rainfallAmount = val;
            rainfallValue.textContent = `${appState.rainfallAmount} mm`;

            // Use requestAnimationFrame for smooth UI updates
            if (appState.rafHandle) cancelAnimationFrame(appState.rafHandle);
            appState.rafHandle = requestAnimationFrame(async () => {
                const village = appState.data.villages[appState.currentVillageId];
                updateSimulationImpact();

                // Fetch from API and update map
                await fetchFloodSimulation();
                updateMapVision(village);
                startFloodAnimation();
            });
        });
    }

    // 5. Close Report Button
    const closeReportBtn = document.getElementById('closeReport');
    if (closeReportBtn) {
        closeReportBtn.addEventListener('click', () => {
            document.getElementById('simulationReport').style.display = 'none';
        });
    }
}

function updateTimeUI(idx) {
    const ticks = document.querySelectorAll('.tick');
    ticks.forEach(t => t.classList.remove('active'));
    if (ticks[idx]) ticks[idx].classList.add('active');

    const labels = ['+0h', '+4h', '+8h', '+12h', '+16h', '+20h', '24h'];
    document.getElementById('timeDisplay').textContent = labels[idx];
    document.getElementById('timelineFill').style.width = `${(idx / 6) * 100}%`;
}

/**
 * Toggle layer visibility on button click
 * Called directly from onclick handlers in HTML
 */
function setupLayerToggle(btnId, layerId) {
    const btn = document.getElementById(btnId);
    if (!btn) return;

    const isActive = btn.classList.contains('active');

    // Toggle button state
    if (isActive) {
        btn.classList.remove('active');
    } else {
        btn.classList.add('active');
    }

    // Apply layer visibility change
    if (appState.map && appState.map.getStyle()) {
        const village = appState.data?.villages[appState.currentVillageId];

        // Handle population density layer
        if (layerId === 'population-layer') {
            togglePopulationHeatmap(!isActive);
            return;
        }

        // Handle soil saturation layer
        if (layerId === 'soil-saturation-layer') {
            if (!isActive && village) {
                generateSoilGrid(village);
            }
            if (appState.map.getLayer('soil-saturation-layer')) {
                appState.map.setLayoutProperty('soil-saturation-layer', 'visibility', !isActive ? 'visible' : 'none');
            }
            return;
        }

        // Handle flood risk layer
        if (layerId === 'flood-risk-layer') {
            const currentLayerId = `flood-risk-layer-${appState.currentTimeStep}`;
            if (appState.map.getLayer(currentLayerId)) {
                appState.map.setLayoutProperty(currentLayerId, 'visibility', !isActive ? 'visible' : 'none');
            }
            if (village) {
                updateMapVision(village);
            }
            return;
        }

        // Handle other static layers
        if (appState.map.getLayer(layerId)) {
            appState.map.setLayoutProperty(layerId, 'visibility', !isActive ? 'visible' : 'none');
        }
    }
}

/**
 * Toggle population density heatmap visibility
 */
function togglePopulationHeatmap(show) {
    if (!appState.map) return;

    // Ensure population data is rendered
    if (show && appState.apiData.population) {
        renderAPIPopulation();
    }

    // Toggle API population layer
    if (appState.map.getLayer('api-population-layer')) {
        appState.map.setLayoutProperty('api-population-layer', 'visibility', show ? 'visible' : 'none');
    }

    // Also toggle legacy population layer if it exists
    if (appState.map.getLayer('population-layer')) {
        appState.map.setLayoutProperty('population-layer', 'visibility', show ? 'visible' : 'none');
    }

    // Show toast notification
    if (show) {
        showToast('Population Density', 'Showing population at risk - red areas indicate higher density', 'info');
    }
}

function updateSimulationImpact() {
    if (!appState.map) return;

    // 1. Update Flood Layers Risk Intensity
    const rainfallFactor = Math.min(1.0, Math.max(0.2, appState.rainfallAmount / 200));
    const timeSteps = ['0h', '4h', '8h', '12h', '16h', '20h', '24h'];

    timeSteps.forEach(ts => {
        const layerId = `flood-risk-layer-${ts}`;
        if (appState.map.getLayer(layerId) && ts === appState.currentTimeStep) {
            appState.map.setPaintProperty(layerId, 'fill-opacity', 0.8 * rainfallFactor);
        }
    });

    // 2. Update Rescue Path (if active) and Report Data
    // Auto-Show Rescue Path if High Risk
    if (appState.rainfallAmount > 100) {
        if (!document.getElementById('btnLayerRescue')?.classList.contains('active')) {
            document.getElementById('btnLayerRescue')?.classList.add('active');
            const village = appState.data.villages[appState.currentVillageId];
            generateRescuePath(village);
            // Notification removed as per user request
        }
    }

    const isRescueVisible = document.getElementById('btnLayerRescue')?.classList.contains('active');
    if (isRescueVisible) {
        // Debounce heavy pathfinding to avoid slider lag
        if (appState.rescueDebounce) clearTimeout(appState.rescueDebounce);
        appState.rescueDebounce = setTimeout(() => {
            const village = appState.data.villages[appState.currentVillageId];
            generateRescuePath(village);
        }, 50);
    }
}

function toggleCompareMode(active) {
    const btn = document.getElementById('btnLayerCompare');
    if (!btn) return;

    const layerId = `flood-risk-layer-${appState.currentTimeStep}`;

    if (active) {
        // Impact mode: High contrast
        if (appState.map.getLayer(layerId)) {
            appState.map.setPaintProperty(layerId, 'fill-opacity', 0.95);
        }
        appState.map.setLayoutProperty('3d-buildings', 'visibility', 'none');
    } else {
        // Normal mode
        if (appState.map.getLayer(layerId)) {
            appState.map.setPaintProperty(layerId, 'fill-opacity', 0.8);
        }
        const is3DActive = document.getElementById('btnLayerBuild')?.classList.contains('active');
        if (appState.map.getLayer('3d-buildings')) {
            appState.map.setLayoutProperty('3d-buildings', 'visibility', is3DActive ? 'visible' : 'none');
        }
    }
}

// ============================================
// Rescue Path System (Google Maps API Integration)
// ============================================

// Stub function to prevent reference errors (rescue path logic was removed)
function generateRescuePath(village) {
    // No-op: Rescue path functionality disabled
    console.log('Rescue path generation skipped (feature disabled)');
}

// ============================================
// Simulation Report Generator
// ============================================

function generateSimulationReport(village) {
    const rainfall = appState.rainfallAmount;
    const timeStep = appState.currentTimeStep;

    // Impact stats from API if available
    let affectedPop = 0;
    let floodSeverity = 'LOW';
    let severityColor = '#22c55e';
    let maxDepth = (rainfall / 50).toFixed(1);

    if (appState.apiData.floodSimulation) {
        const currentSim = appState.apiData.floodSimulation.features.find(f => f.properties.timestep.includes(timeStep));
        if (currentSim) {
            maxDepth = currentSim.properties.max_depth_m.toFixed(1);
            floodSeverity = currentSim.properties.severity.toUpperCase();
        }
    }

    // Realistic Impact Calculation
    const basePop = village.info.population || 10000;
    const riskFactor = Math.min(1.0, rainfall / 300);
    affectedPop = Math.round(basePop * (riskFactor * 0.4 + 0.1));

    if (floodSeverity === 'EXTREME') severityColor = '#7f1d1d';
    else if (floodSeverity === 'HIGH') severityColor = '#dc2626';
    else if (floodSeverity === 'MODERATE') severityColor = '#eab308';

    // Rescue Logistics
    const evacTimeHours = (affectedPop / 2000).toFixed(1);
    const sheltersActive = Math.ceil(affectedPop / 500);
    const ndrfTeams = Math.ceil(affectedPop / 1000);

    const reportHTML = `
        <div style="margin-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;">
            <div style="font-size:1.1rem; font-weight:700; color:#fff; display:flex; justify-content:space-between; align-items:center;">
                <span>SITUATION REPORT</span>
                <span style="font-size:0.8rem; background:${severityColor}; padding:2px 8px; border-radius:4px;">${floodSeverity}</span>
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">
                ZONE: ${village.info.name.toUpperCase()} &bull; RAINFALL: ${rainfall}mm &bull; MAX DEPTH: ${maxDepth}m
            </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:15px;">
             <div class="stat-box" style="background:rgba(255,255,255,0.03); padding:8px; border-radius:6px;">
                <div style="color:var(--text-muted); font-size:0.7rem;">AFFECTED POPULATION</div>
                <div style="font-size:1.2rem; font-weight:700; color:#ef4444;">${affectedPop.toLocaleString()}</div>
             </div>
             <div class="stat-box" style="background:rgba(255,255,255,0.03); padding:8px; border-radius:6px;">
                <div style="color:var(--text-muted); font-size:0.7rem;">EST. EVACUATION TIME</div>
                <div style="font-size:1.2rem; font-weight:700; color:#eab308;">${evacTimeHours} HRS</div>
             </div>
        </div>
        
        <div style="margin-bottom:15px;">
            <div style="font-size:0.8rem; font-weight:600; color:#22d3ee; margin-bottom:8px; border-bottom:1px solid #22d3ee33; padding-bottom:4px;">
                RESCUE LOGISTICS & RESOURCES
            </div>
            <div style="font-size:0.75rem; display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
                <div>⛑️ NDRF Teams: <strong style="color:#fff">${ndrfTeams}</strong></div>
                <div>⛺ Shelters Active: <strong style="color:#fff">${sheltersActive}</strong></div>
                <div>🚁 Heli-drop Zones: <strong style="color:#fff">${rainfall > 150 ? 2 : 0}</strong></div>
                <div>🚤 Rescue Boats: <strong style="color:#fff">${Math.max(5, Math.floor(rainfall / 5))}</strong></div>
            </div>
        </div>
    `;

    const reportEl = document.getElementById('simulationReport');
    const contentEl = document.getElementById('reportContent');

    if (reportEl && contentEl) {
        contentEl.innerHTML = reportHTML;
        reportEl.style.display = 'block';
    }
}



function getRecommendedAction(severity) {
    switch (severity) {
        case 'CRITICAL': return 'IMMEDIATE EVACUATION';
        case 'HIGH': return 'Prepare for evacuation';
        case 'MODERATE': return 'Monitor conditions';
        default: return 'Normal operations';
    }
}

async function run() {
    // Global safety: Always hide loading after 5 seconds regardless of network
    const globalTimeout = setTimeout(() => {
        console.warn("Global Startup Timeout: Forcing UI visibility");
        hideLoading();
    }, 5000);

    const success = await fetchDashboardData();
    if (success) {
        clearTimeout(globalTimeout);
        bindEvents();
        init3DMap();

        // Start Mission Clock
        setInterval(updateClock, 1000);
        updateClock();

        // Start Synthetic Data Updates (temperature, soil) - updates every 60 seconds
        setInterval(updateSyntheticData, 60000);
        updateSyntheticData();
    }
}

// ============================================
// Synthetic Data Generator
// ============================================

// Realistic CURRENT temperatures for January in India
// Based on actual climate data for these regions
const VILLAGE_CLIMATE = {
    'wayanad_meppadi': { baseTemp: 22.0, tempFluctuation: 0.3, baseSoil: 0.35, soilFluctuation: 0.005 },  // Kerala hill station - cool
    'darbhanga': { baseTemp: 18.5, tempFluctuation: 0.4, baseSoil: 0.22, soilFluctuation: 0.008 },       // Bihar plains - winter cold
    'dhemaji': { baseTemp: 16.0, tempFluctuation: 0.5, baseSoil: 0.30, soilFluctuation: 0.006 }          // Assam - winter cool
};

// Track current values for smooth transitions
let currentTempValues = {};
let currentSoilValues = {};

async function updateSyntheticData() {
    const village = appState.data?.villages[appState.currentVillageId];
    if (!village) return;

    const villageId = appState.currentVillageId;
    const coords = village.info.coordinates;

    // Fetch REAL weather instead of synthetic base values
    try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${coords.lat}&longitude=${coords.lon}&current=temperature_2m,relative_humidity_2m&timezone=auto`;
        const response = await fetch(url);
        const weatherData = await response.json();

        if (weatherData && weatherData.current) {
            const temperature = weatherData.current.temperature_2m;
            // Soil moisture is harder to get live for free, so we use a realistic stable value for Jan
            const baseSoil = VILLAGE_CLIMATE[villageId]?.baseSoil || 0.35;
            const soilMoisture = baseSoil + (Math.random() - 0.5) * 0.01;

            // Update UI
            const safeUpdate = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.textContent = val;
            };

            safeUpdate('valTemp', `${temperature.toFixed(1)}°C`);
            safeUpdate('valSoil', `${(soilMoisture * 100).toFixed(1)}%`);

            console.log(`Live Weather Update [${villageId}]: ${temperature}°C`);
        }
    } catch (e) {
        console.error("Live weather fetch failed, using fallback:", e);
    }
}

function updateClock() {
    const clock = document.getElementById('missionClock');
    if (clock) {
        const now = new Date();
        // Format: HH:MM:SS UTC
        const time = now.toISOString().substring(11, 19);
        clock.textContent = `${time} UTC`;

        // Random "Telemetry" effect occasionally
        if (Math.random() < 0.05) {
            clock.style.color = '#ef4444'; // Red flash
            setTimeout(() => clock.style.color = '', 200);
        }
    }
}

// ============================================
// Jal Drishti Decision Support Systems (DSS)
// ============================================

let voiceAlertTriggered = false;

function checkRiskForVoiceAlert(village) {
    if (voiceAlertTriggered) return;

    const stats = village.statistics?.['24h'] || {};
    const riskAreaHigh = stats.area_at_risk_km2?.high || 0;

    if (riskAreaHigh > 1.0) { // If more than 1km2 is high risk
        triggerVoiceAlert(village.info.name);
        voiceAlertTriggered = true;
        // Reset after 30 seconds to allow re-triggering if conditions persist
        setTimeout(() => { voiceAlertTriggered = false; }, 30000);
    }
}

function triggerVoiceAlert(cityName) {
    if (!('speechSynthesis' in window)) return;

    const msg = new SpeechSynthesisUtterance();
    msg.text = `Warning. Flood levels critical in ${cityName}. Evacuate to safe zone immediately.`;
    msg.pitch = 1.2;
    msg.rate = 0.9;
    window.speechSynthesis.speak(msg);

    // Also show visual alert
    showToast("CRITICAL FLOOD ALERT", `High risk detected in ${cityName}. Evacuation routes active.`, "error");
}

function showToast(title, message, type = 'info') {
    const container = document.getElementById('toast-container') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `glass-panel toast toast-${type}`;
    toast.style.cssText = `
        padding: 15px; margin-bottom: 10px; border-left: 4px solid ${type === 'error' ? '#ef4444' : '#22d3ee'};
        background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px); color: #fff;
        animation: slideIn 0.3s ease-out;
    `;
    toast.innerHTML = `<strong>${title}</strong><div style="font-size:12px; opacity:0.8">${message}</div>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function createToastContainer() {
    const div = document.createElement('div');
    div.id = 'toast-container';
    div.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column;';
    document.body.appendChild(div);
    return div;
}

// Draggable Infrastructure Logic Removed

// ============================================
// RESCUE MODE SYSTEM
// ============================================

/**
 * Toggle rescue mode - when active, clicking map triggers rescue path calculation
 */
function toggleRescueMode() {
    appState.rescueMode = !appState.rescueMode;
    // Support both navbar button and layer overlay button
    const btnNavbar = document.getElementById('btnRescueMode');
    const btnLayer = document.getElementById('btnFindRescue');

    if (appState.rescueMode) {
        // Activate both buttons if they exist
        [btnNavbar, btnLayer].forEach(btn => {
            if (btn) {
                btn.classList.add('active');
                btn.style.background = 'rgba(239, 68, 68, 0.3)';
                btn.style.borderColor = '#ef4444';
            }
        });
        if (btnNavbar) btnNavbar.innerHTML = '⛑️ RESCUE MODE ON';
        if (btnLayer) btnLayer.innerHTML = '<span class="layer-icon">🛟</span> CLICK MAP TO SET LOCATION';

        // Change cursor
        if (appState.map) {
            appState.map.getCanvas().style.cursor = 'crosshair';
        }

        showToast('Rescue Mode Active', 'Click anywhere on the map to find the safest evacuation route.', 'info');

        // Add map click handler
        appState.map.on('click', handleRescueClick);
    } else {
        // Deactivate both buttons
        [btnNavbar, btnLayer].forEach(btn => {
            if (btn) {
                btn.classList.remove('active');
                btn.style.background = '';
                btn.style.borderColor = '';
            }
        });
        if (btnNavbar) btnNavbar.innerHTML = '⛑️ RESCUE ME';
        if (btnLayer) btnLayer.innerHTML = '<span class="layer-icon">🛟</span> Find Best Rescue Path';

        if (appState.map) {
            appState.map.getCanvas().style.cursor = '';
            appState.map.off('click', handleRescueClick);
        }

        // Clear rescue path
        clearRescuePath();
    }
}

/**
 * Handle map click in rescue mode
 */
async function handleRescueClick(e) {
    if (!appState.rescueMode) return;

    const { lng, lat } = e.lngLat;
    showToast('Calculating Route...', `Tactical pathfinding from (${lat.toFixed(4)}, ${lng.toFixed(4)})`, 'info');

    try {
        // Try API first
        let result = await fetchFromAPI('/api/rescue', {
            lat: lat,
            lon: lng,
            village_id: appState.currentVillageId,
            flood_level: appState.rainfallAmount / 100
        });

        if (!result || result.status !== 'success') {
            console.log("Backend routing failed or unavailable, using client-side fallback...");
            result = calculateClientSideRescueRoute(lng, lat);
        }

        if (result && result.status === 'success') {
            drawRescuePath(result.rescue_path);

            const summary = result.rescue_path.summary;
            showToast(
                '✅ Safe Route Found!',
                `${summary.distance_km}km to ${summary.destination} (~${summary.estimated_time_min} min walk)`,
                'info'
            );
        } else {
            showToast('⚠️ No Safe Route', 'Area may be isolated. Seek highest available ground immediately!', 'error');
        }
    } catch (e) {
        console.error('Rescue route calculation failed:', e);
        // Fallback to client-side even on catch
        const fallback = calculateClientSideRescueRoute(lng, lat);
        if (fallback.status === 'success') {
            drawRescuePath(fallback.rescue_path);
            showToast('✅ Tactical Route Found', 'Using local fallback routing.', 'info');
        } else {
            showToast('Route Error', 'Could not determine safe passage.', 'error');
        }
    }
}

/**
 * Client-side fallback for rescue routing
 * Works for all villages by finding the nearest infrastructure of type 'hospital' or 'shelter'
 */
function calculateClientSideRescueRoute(userLng, userLat) {
    const config = SIMULATION_CONFIG[appState.currentVillageId] || SIMULATION_CONFIG.wayanad_meppadi;

    // Define safe havens based on village (hardcoded fallbacks for all 3 major villages)
    const havens = {
        'wayanad_meppadi': [
            { name: 'Meppadi Primary Health Centre', lat: 11.558, lon: 76.132 },
            { name: 'St. Joseph Community Shelter', lat: 11.552, lon: 76.141 }
        ],
        'darbhanga': [
            { name: 'DMCH Hospital Safe Zone', lat: 26.126, lon: 85.895 },
            { name: 'North Bihar Relief Camp', lat: 26.115, lon: 85.908 }
        ],
        'dhemaji': [
            { name: 'Dhemaji Civil Hospital', lat: 27.485, lon: 94.555 },
            { name: 'Flood Relief Shelter A', lat: 27.475, lon: 94.568 }
        ]
    };

    const villageHavens = havens[appState.currentVillageId] || havens.wayanad_meppadi;

    // Find nearest haven
    let nearest = villageHavens[0];
    let minDist = Infinity;

    villageHavens.forEach(h => {
        const d = Math.sqrt(Math.pow(h.lat - userLat, 2) + Math.pow(h.lon - userLng, 2));
        if (d < minDist) {
            minDist = d;
            nearest = h;
        }
    });

    // Generate a jagged "synthetic" path to simulate road navigation
    const steps = 8;
    const coordinates = [[userLng, userLat]];

    for (let i = 1; i < steps; i++) {
        const ratio = i / steps;
        const baseLng = userLng + (nearest.lon - userLng) * ratio;
        const baseLat = userLat + (nearest.lat - userLat) * ratio;

        // Add random "jitter" to simulate road turns
        const jitter = 0.0005;
        coordinates.push([
            baseLng + (Math.random() - 0.5) * jitter,
            baseLat + (Math.random() - 0.5) * jitter
        ]);
    }
    coordinates.push([nearest.lon, nearest.lat]);

    const distanceKm = (minDist * 111.32).toFixed(2); // Rough degree to km
    const timeMin = Math.round(distanceKm * 15); // Avg 4km/h walking speed

    return {
        status: 'success',
        rescue_path: {
            summary: {
                distance_km: distanceKm,
                destination: nearest.name,
                estimated_time_min: timeMin
            },
            features: [
                {
                    type: 'Feature',
                    properties: { type: 'rescue_path' },
                    geometry: { type: 'LineString', coordinates: coordinates }
                },
                {
                    type: 'Feature',
                    properties: { type: 'user_location' },
                    geometry: { type: 'Point', coordinates: [userLng, userLat] }
                },
                {
                    type: 'Feature',
                    properties: { type: 'safe_haven' },
                    geometry: { type: 'Point', coordinates: [nearest.lon, nearest.lat] }
                }
            ]
        }
    };
}

/**
 * Draw rescue path on map
 */
function drawRescuePath(pathData) {
    if (!appState.map || !pathData) return;

    clearRescuePath();

    const pathFeature = pathData.features?.find(f => f.properties?.type === 'rescue_path');
    const startMarker = pathData.features?.find(f => f.properties?.type === 'user_location');
    const endMarker = pathData.features?.find(f => f.properties?.type === 'safe_haven');

    // Add rescue path source and layer
    if (pathFeature) {
        appState.map.addSource('rescue-path-source', {
            type: 'geojson',
            data: pathFeature
        });

        // Glow layer
        appState.map.addLayer({
            id: 'rescue-path-glow',
            type: 'line',
            source: 'rescue-path-source',
            paint: {
                'line-color': '#22c55e',
                'line-width': 12,
                'line-opacity': 0.3,
                'line-blur': 4
            }
        });

        // Main path layer with animation
        appState.map.addLayer({
            id: 'rescue-path-layer',
            type: 'line',
            source: 'rescue-path-source',
            paint: {
                'line-color': '#22c55e',
                'line-width': 4,
                'line-opacity': 1,
                'line-dasharray': [2, 1]
            }
        });
    }

    // Add markers
    if (startMarker) {
        appState.map.addSource('rescue-start-source', {
            type: 'geojson',
            data: startMarker
        });
        appState.map.addLayer({
            id: 'rescue-start-layer',
            type: 'circle',
            source: 'rescue-start-source',
            paint: {
                'circle-radius': 10,
                'circle-color': '#3b82f6',
                'circle-stroke-width': 3,
                'circle-stroke-color': '#fff'
            }
        });
    }

    if (endMarker) {
        appState.map.addSource('rescue-end-source', {
            type: 'geojson',
            data: endMarker
        });
        appState.map.addLayer({
            id: 'rescue-end-layer',
            type: 'circle',
            source: 'rescue-end-source',
            paint: {
                'circle-radius': 12,
                'circle-color': '#22c55e',
                'circle-stroke-width': 3,
                'circle-stroke-color': '#fff'
            }
        });
    }

    appState.rescuePath = pathData;
}

/**
 * Clear rescue path from map
 */
function clearRescuePath() {
    if (!appState.map) return;

    const layers = ['rescue-path-glow', 'rescue-path-layer', 'rescue-start-layer', 'rescue-end-layer'];
    const sources = ['rescue-path-source', 'rescue-start-source', 'rescue-end-source'];

    layers.forEach(id => {
        if (appState.map.getLayer(id)) appState.map.removeLayer(id);
    });

    sources.forEach(id => {
        if (appState.map.getSource(id)) appState.map.removeSource(id);
    });

    appState.rescuePath = null;
}

// ============================================
// FLOOD ANIMATION ENGINE
// ============================================

/**
 * Animate flood polygons with pulsing opacity and T1→T2→T3 transitions
 */
function startFloodAnimation() {
    if (appState.floodAnimationId) return;

    let frame = 0;
    const timesteps = ['t1', 't2', 't3'];
    let currentTimestepIndex = 0;

    appState.floodAnimationId = setInterval(() => {
        frame++;

        // Pulse opacity (0.4 to 0.8)
        const pulse = 0.4 + 0.4 * Math.sin(frame * 0.15);

        // Update flood layer opacity
        const layerId = `flood-risk-layer-${appState.currentTimeStep}`;
        if (appState.map && appState.map.getLayer(layerId)) {
            appState.map.setPaintProperty(layerId, 'fill-opacity', pulse);
        }

        // Switch timesteps every 60 frames (~2 seconds)
        if (frame % 60 === 0 && appState.apiData.floodSimulation) {
            currentTimestepIndex = (currentTimestepIndex + 1) % timesteps.length;
            // Could switch flood polygon data here if using API-driven simulation
        }
    }, 33); // ~30fps
}

/**
 * Stop flood animation
 */
function stopFloodAnimation() {
    if (appState.floodAnimationId) {
        clearInterval(appState.floodAnimationId);
        appState.floodAnimationId = null;
    }
}

/**
 * Fetch flood simulation from API
 */
async function fetchFloodSimulation() {
    const result = await fetchFromAPI('/api/simulate', {
        rainfall: appState.rainfallAmount,
        village_id: appState.currentVillageId,
        format: 'polygons'
    });

    if (result && result.status === 'success') {
        appState.apiData.floodSimulation = result.simulation;
        console.log('✓ Flood simulation loaded from API');
        return result.simulation;
    }
    return null;
}

// ============================================
// VILLAGE BOUNDARY RENDERING (Glowing Dashed)
// ============================================

/**
 * Render village boundary with glowing dashed orange/white style
 */
function renderAPIBoundary() {
    if (!appState.map || !appState.apiData.boundary) return;

    const boundary = appState.apiData.boundary;

    // Remove existing if present
    if (appState.map.getLayer('api-boundary-glow')) {
        appState.map.removeLayer('api-boundary-glow');
        appState.map.removeLayer('api-boundary-line');
        appState.map.removeSource('api-boundary-source');
    }

    appState.map.addSource('api-boundary-source', {
        type: 'geojson',
        data: boundary
    });

    // Glow effect (white blur)
    appState.map.addLayer({
        id: 'api-boundary-glow',
        type: 'line',
        source: 'api-boundary-source',
        paint: {
            'line-color': '#ffffff',
            'line-width': 10,
            'line-opacity': 0.25,
            'line-blur': 6
        }
    });

    // Main boundary (orange dashed)
    appState.map.addLayer({
        id: 'api-boundary-line',
        type: 'line',
        source: 'api-boundary-source',
        paint: {
            'line-color': '#ff9500',
            'line-width': 4,
            'line-opacity': 0.9,
            'line-dasharray': [2, 1]
        }
    });
}

/**
 * Render infrastructure POIs on map
 */
function renderAPIPOIs() {
    if (!appState.map || !appState.apiData.infrastructure) return;

    const pois = appState.apiData.infrastructure;

    if (appState.map.getLayer('api-pois-layer')) {
        appState.map.removeLayer('api-pois-labels');
        appState.map.removeLayer('api-pois-layer');
        appState.map.removeSource('api-pois-source');
    }

    appState.map.addSource('api-pois-source', {
        type: 'geojson',
        data: pois
    });

    appState.map.addLayer({
        id: 'api-pois-layer',
        type: 'circle',
        source: 'api-pois-source',
        paint: {
            'circle-radius': ['case', ['get', 'is_safe_haven'], 12, 8],
            'circle-color': ['case', ['get', 'is_safe_haven'], '#22c55e', '#ef4444'],
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff'
        }
    });

    appState.map.addLayer({
        id: 'api-pois-labels',
        type: 'symbol',
        source: 'api-pois-source',
        layout: {
            'text-field': ['get', 'name'],
            'text-size': 11,
            'text-offset': [0, 1.5],
            'text-anchor': 'top'
        },
        paint: {
            'text-color': '#ffffff',
            'text-halo-color': '#000000',
            'text-halo-width': 1
        }
    });
}

/**
 * Render population heatmap from API data
 * Shows population density to identify populations at risk
 */
function renderAPIPopulation() {
    if (!appState.map || !appState.apiData.population) return;

    const population = appState.apiData.population;
    
    // Check if button is active to determine initial visibility
    const isPopBtnActive = document.getElementById('btnLayerPop')?.classList.contains('active');
    const visibility = isPopBtnActive ? 'visible' : 'none';

    if (appState.map.getSource('api-population-source')) {
        appState.map.getSource('api-population-source').setData(population);
    } else {
        appState.map.addSource('api-population-source', {
            type: 'geojson',
            data: population
        });

        appState.map.addLayer({
            id: 'api-population-layer',
            type: 'heatmap',
            source: 'api-population-source',
            layout: { visibility: visibility },
            paint: {
                'heatmap-weight': ['get', 'intensity'],
                'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 14, 3],
                'heatmap-color': [
                    'interpolate', ['linear'], ['heatmap-density'],
                    0, 'rgba(50, 50, 150, 0)',       // Transparent for low density
                    0.15, 'rgba(0, 100, 200, 0.4)',  // Blue for low population
                    0.3, 'rgba(0, 180, 200, 0.5)',   // Cyan for moderate
                    0.5, 'rgba(100, 200, 100, 0.6)', // Green for medium
                    0.7, 'rgba(255, 200, 0, 0.75)', // Yellow/Orange for high
                    0.85, 'rgba(255, 100, 50, 0.85)', // Orange-red for very high
                    1, 'rgba(255, 0, 0, 0.95)'       // Red for highest density (most at risk)
                ],
                'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 
                    10, 15,   // Smaller radius when zoomed out
                    13, 30,   // Medium radius at default zoom
                    16, 50    // Larger radius when zoomed in
                ],
                'heatmap-opacity': 0.8
            }
        });
    }
    
    // Ensure visibility matches button state
    if (appState.map.getLayer('api-population-layer')) {
        appState.map.setLayoutProperty('api-population-layer', 'visibility', visibility);
    }
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    @keyframes slideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }
    @keyframes pulse-red { 0% { transform: scale(1); filter: drop-shadow(0 0 0 red); } 50% { transform: scale(1.2); filter: drop-shadow(0 0 15px red); } 100% { transform: scale(1); filter: drop-shadow(0 0 0 red); } }
    @keyframes rescue-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    
    #btnRescueMode {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: var(--font-tech);
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    #btnRescueMode:hover {
        background: rgba(239, 68, 68, 0.3);
        border-color: #ef4444;
        transform: scale(1.05);
    }
    
    #btnRescueMode.active {
        animation: rescue-pulse 1.5s infinite;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
`;
document.head.appendChild(style);

// --- CITY-SPECIFIC SIMULATION CONFIGURATION ---

const SIMULATION_CONFIG = {
    wayanad_meppadi: {
        type: 'hilly_landslide',
        name: 'Western Ghats Hilly',
        baseElevation: 780,
        elevationRange: 200,
        runoffMultiplier: 1.8,
        flashFloodProne: true,
        riskFactors: ['Landslide Risk', 'Flash Flood', 'Debris Flow'],
        hazardIcon: '⛰️',
        floodCharacteristic: 'Valley accumulation with steep runoff channels',
        evacuationAdvice: 'Move to higher ground away from valleys and slopes',
        bbox: [76.10, 11.52, 76.17, 11.59] // Refined for better resolution
    },
    darbhanga: {
        type: 'riverine_plain',
        name: 'Gangetic Plain',
        baseElevation: 53,
        elevationRange: 10,
        runoffMultiplier: 0.8,
        embankmentBreachProne: true,
        riskFactors: ['River Overflow', 'Embankment Breach', 'Waterlogging'],
        hazardIcon: '🌊',
        floodCharacteristic: 'Linear river channel flooding with slow lateral spread',
        evacuationAdvice: 'Move away from river embankments to designated shelters',
        bbox: [85.85, 26.12, 85.93, 26.19] // Refined for better resolution
    },
    dhemaji: {
        type: 'floodplain',
        name: 'Brahmaputra Floodplain',
        baseElevation: 53,
        elevationRange: 15,
        runoffMultiplier: 1.2,
        riverSwellProne: true,
        riskFactors: ['River Swell', 'Bank Erosion', 'Sheet Flooding'],
        hazardIcon: '🏞️',
        floodCharacteristic: 'Wide-area sheet flooding with tributary confluence',
        evacuationAdvice: 'Move to elevated platforms or community shelters on high ground',
        bbox: [94.53, 27.45, 94.60, 27.51] // Refined for better resolution
    }
};

// --- LOCATION SPECIFIC SIMULATION ALGORITHMS ---

/**
 * WAYANAD (Western Ghats) - Landslide-prone hilly terrain
 * Characteristics: Sharp valley flooding, steep runoff channels, debris flow corridors
 */
function generateHillyFlood(x, y, gridSize, intensity) {
    const dx = (x - gridSize / 2);
    const dy = (y - gridSize / 2);
    const dist = Math.sqrt(dx * dx + dy * dy) / (gridSize / 2.5);

    // Multiple valley channels radiating from center (landslide corridors)
    const valley1 = Math.exp(-Math.pow(dy - dx * 0.3, 2) / 8);  // NE-SW valley
    const valley2 = Math.exp(-Math.pow(dy + dx * 0.2 - 5, 2) / 6);  // Secondary valley
    const valley3 = Math.exp(-Math.pow(dx - 10, 2) / 10);  // Vertical stream

    // Steep slope accumulation zones
    const slopeAccum = Math.sin(x * 0.15) * Math.sin(y * 0.12) * 0.6;

    // Flash flood surge pattern - concentrated in low points
    const flashSurge = (1.5 - dist) * (valley1 + valley2 * 0.7 + valley3 * 0.5);

    // Debris flow noise (irregular patterns)
    const debrisNoise = Math.sin(x * 1.2 + y * 0.8) * Math.cos(x * 0.5 - y * 1.1) * 0.4;

    const val = flashSurge + slopeAccum + debrisNoise * intensity;
    return Math.max(0, val * intensity * 5.5);
}

/**
 * DARBHANGA (Gangetic Plain) - Riverine flooding
 * Characteristics: Kosi/Kamla river patterns, embankment breach simulation, slow lateral spread
 */
function generateRiverineFlood(x, y, gridSize, intensity) {
    const centerX = gridSize / 2;
    const centerY = gridSize / 2;

    // Primary river channel (Kosi-like meandering)
    const riverMeander = Math.sin(y * 0.15) * 4;
    const mainRiverDist = Math.abs(x - centerX - riverMeander);
    const mainRiver = Math.exp(-Math.pow(mainRiverDist, 2) / 20);

    // Secondary tributary (Kamla-like)
    const tributaryOffset = Math.sin(y * 0.2 + 2) * 3;
    const tributaryDist = Math.abs(x - centerX + 12 - tributaryOffset);
    const tributary = Math.exp(-Math.pow(tributaryDist, 2) / 15) * 0.7;

    // Embankment breach simulation (localized high-intensity zones)
    const breach1 = Math.exp(-((x - centerX - 5) ** 2 + (y - centerY + 8) ** 2) / 25) * 1.5;
    const breach2 = Math.exp(-((x - centerX + 3) ** 2 + (y - centerY - 10) ** 2) / 30) * 1.2;

    // Slow lateral spreading on flat terrain (waterlogging zones)
    const waterlog = Math.sin(x * 0.08) * Math.sin(y * 0.08) * 0.3;

    // Combine patterns
    const val = (mainRiver * 2.5 + tributary * 1.8 + breach1 + breach2) * (1 + waterlog);
    return Math.max(0, val * intensity * 3.5);
}

/**
 * DHEMAJI (Brahmaputra Floodplain) - Wide-area sheet flooding
 * Characteristics: River swell, multiple tributary confluence, erosion-prone banks
 */
function generateFloodplainFlood(x, y, gridSize, intensity) {
    const centerX = gridSize / 2;
    const centerY = gridSize / 2;

    // Brahmaputra main channel (wide, braided pattern)
    const braidOffset1 = Math.sin(y * 0.12) * 6;
    const braidOffset2 = Math.sin(y * 0.18 + 1.5) * 4;
    const mainChannel = Math.exp(-Math.pow(x - centerX - braidOffset1, 2) / 50) +
        Math.exp(-Math.pow(x - centerX - braidOffset2 - 8, 2) / 40) * 0.6;

    // Multiple tributary confluence (Subansiri, Ranganadi patterns)
    const trib1 = Math.exp(-Math.pow(y - x * 0.4 - 10, 2) / 35) * 0.8;
    const trib2 = Math.exp(-Math.pow(y + x * 0.3 - 25, 2) / 40) * 0.6;

    // Wide sheet flooding (characteristic of floodplains)
    const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2) / (gridSize / 1.8);
    const sheetFlood = Math.max(0, (1.2 - dist)) * 0.8;

    // Bank erosion zones (irregular edge patterns)
    const erosionNoise = (Math.sin(x * 0.3 + y * 0.2) + Math.cos(x * 0.15 - y * 0.25)) * 0.25;

    // Silt deposit accumulation (low-lying areas)
    const siltDeposit = Math.sin(x * 0.05) * Math.sin(y * 0.06) * 0.2;

    const val = mainChannel * 1.8 + trib1 + trib2 + sheetFlood + erosionNoise + siltDeposit;
    return Math.max(0, val * intensity * 3.2);
}

// Legacy function renamed for backward compatibility
function generateUrbanFlood(x, y, gridSize, intensity) {
    return generateFloodplainFlood(x, y, gridSize, intensity);
}

/**
 * Master flood grid generator - routes to appropriate city-specific algorithm
 */
function generateFloodGrid(center, intensity, locationId = 'wayanad') {
    // Debug logging
    console.log(`Generating flood grid for ${locationId}: Intensity ${intensity}`);

    if (intensity <= 0.001) { // Lowered threshold from 0.05
        console.warn("Intensity too low, returning empty grid");
        return { type: 'FeatureCollection', features: [] };
    }

    const features = [];
    const config = SIMULATION_CONFIG[locationId] || SIMULATION_CONFIG.wayanad_meppadi;
    const bbox = config.bbox;

    const minLon = bbox[0];
    const minLat = bbox[1];
    const maxLon = bbox[2];
    const maxLat = bbox[3];

    const gridSize = 40; // Maintain reasonable grid size for performance
    const stepLon = (maxLon - minLon) / gridSize;
    const stepLat = (maxLat - minLat) / gridSize;

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const lon = minLon + (x * stepLon);
            const lat = minLat + (y * stepLat);

            let val = 0;
            let terrainElevation = config.baseElevation;

            if (locationId.includes('darbhanga')) {
                val = generateRiverineFlood(x, y, gridSize, intensity);
                terrainElevation = config.baseElevation + Math.sin(x * 0.1) * 2 + Math.cos(y * 0.1) * 2;
            } else if (locationId.includes('dhemaji')) {
                val = generateFloodplainFlood(x, y, gridSize, intensity);
                terrainElevation = config.baseElevation + Math.sin(x * 0.08 + y * 0.08) * 5;
            } else {
                val = generateHillyFlood(x, y, gridSize, intensity);
                terrainElevation = config.baseElevation + Math.sin(x * 0.15) * 80 + Math.cos(y * 0.12) * 60;
            }

            val *= config.runoffMultiplier;
            if (val < 0) val = 0;
            if (intensity < 0.2 && val > 0.5) val = 0.5;

            const waterDepth = val * 150 * config.runoffMultiplier;

            features.push({
                type: 'Feature',
                properties: {
                    value: val,
                    water_depth_mm: waterDepth,
                    elevation_m: terrainElevation,
                    risk_level: val > 3.5 ? 'extreme' : (val > 2.5 ? 'high' : (val > 1 ? 'medium' : 'low')),
                    terrain_type: config.type
                },
                geometry: {
                    type: 'Polygon',
                    coordinates: [[
                        [lon, lat],
                        [lon + stepLon, lat],
                        [lon + stepLon, lat + stepLat],
                        [lon, lat + stepLat],
                        [lon, lat]
                    ]]
                }
            });
        }
    }
    return { type: 'FeatureCollection', features: features };
}

/**
 * Enhanced Deep Scan with city-specific insights
 * Shows terrain-aware elevation, location-specific risk factors, population at risk, and evacuation advice
 */
function handleDeepScan(feature, lngLat) {
    const inspector = document.getElementById('cellInspector');
    if (inspector) inspector.style.display = 'block';

    // Get current village configuration
    const locationId = appState.currentVillageId;
    const config = SIMULATION_CONFIG[locationId] || SIMULATION_CONFIG.wayanad_meppadi;

    let props = {};
    if (appState.map) {
        const point = appState.map.project(lngLat);
        const features = appState.map.queryRenderedFeatures(point, {
            layers: [`flood-risk-layer-${appState.currentTimeStep}`]
        });
        if (features && features.length > 0) {
            props = features[0].properties;
        }
    }

    // Use city-specific elevation with terrain-based variations
    let elevation = props.elevation_m;
    if (!elevation) {
        // Generate realistic elevation based on terrain type
        const pseudoRandom = (lngLat.lng * 1000 + lngLat.lat * 1000) % 100;
        elevation = config.baseElevation + (Math.sin(pseudoRandom * 0.1) * config.elevationRange * 0.5);
    }

    const depth = props.water_depth_mm || props.depth || 0;
    const riskLevel = props.risk_level || 'SAFE';
    const riskColor = riskLevel === 'extreme' ? '#ef4444' :
        (riskLevel === 'high' ? '#f97316' :
            (riskLevel === 'medium' ? '#eab308' : '#22c55e'));

    // Calculate population at risk for this tile
    const popAtRisk = calculatePopulationAtRisk(lngLat, riskLevel, depth);

    const safeUpdate = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    // Update standard Deep Scan values
    safeUpdate('inspectElev', `${Math.round(elevation)}m`);
    safeUpdate('inspectDepth', `${parseFloat(depth).toFixed(1)}mm`);
    safeUpdate('inspectPopAtRisk', popAtRisk.count > 0 ? popAtRisk.count.toLocaleString() : '0');
    safeUpdate('inspectDensity', popAtRisk.density);

    // Color the population at risk based on count
    const popEl = document.getElementById('inspectPopAtRisk');
    if (popEl) {
        if (popAtRisk.count > 500) popEl.style.color = '#ef4444';
        else if (popAtRisk.count > 200) popEl.style.color = '#f97316';
        else if (popAtRisk.count > 50) popEl.style.color = '#eab308';
        else popEl.style.color = '#22c55e';
    }

    const riskEl = document.getElementById('inspectRisk');
    if (riskEl) {
        riskEl.textContent = riskLevel.toUpperCase();
        riskEl.style.color = riskColor;
    }

    // Update city-specific info in the panel (if elements exist)
    const terrainTypeEl = document.getElementById('inspectTerrainType');
    if (terrainTypeEl) {
        terrainTypeEl.textContent = `${config.hazardIcon} ${config.name}`;
    }

    const riskFactorsEl = document.getElementById('inspectRiskFactors');
    if (riskFactorsEl && riskLevel !== 'low' && riskLevel !== 'safe') {
        // Show relevant risk factors only when there's actual risk
        const relevantFactors = config.riskFactors.slice(0, riskLevel === 'extreme' ? 3 : (riskLevel === 'high' ? 2 : 1));
        riskFactorsEl.innerHTML = relevantFactors.map(f => `<span class="risk-factor-tag">${f}</span>`).join(' ');
        riskFactorsEl.style.display = 'block';
    } else if (riskFactorsEl) {
        riskFactorsEl.style.display = 'none';
    }

    const evacuationEl = document.getElementById('inspectEvacuation');
    if (evacuationEl && (riskLevel === 'high' || riskLevel === 'extreme')) {
        const urgency = popAtRisk.count > 200 ? 'URGENT: ' : '';
        evacuationEl.textContent = `${urgency}${config.evacuationAdvice}${popAtRisk.count > 100 ? ` Est. ${popAtRisk.count} people need evacuation.` : ''}`;
        evacuationEl.style.display = 'block';
    } else if (evacuationEl) {
        evacuationEl.style.display = 'none';
    }

    // Update risk gradient marker (Tactical UI logic)
    const riskMap = { 'low': 20, 'medium': 50, 'high': 80, 'extreme': 100, 'safe': 5 };
    const markerPos = riskMap[riskLevel.toLowerCase()] || 5;
    const marker = document.querySelector('.risk-marker');
    if (marker) {
        // Bar fills from left to right: left -100% is empty, 0% is full
        marker.style.left = `${markerPos - 100}%`;
    }

    // Update highlight circle on map
    if (appState.map && lngLat) {
        if (!appState.map.getSource('scan-highlight')) {
            appState.map.addSource('scan-highlight', {
                type: 'geojson',
                data: { "type": "Feature", "geometry": { "type": "Point", "coordinates": [lngLat.lng, lngLat.lat] } }
            });
            appState.map.addLayer({
                id: 'scan-highlight-layer',
                type: 'circle',
                source: 'scan-highlight',
                paint: {
                    'circle-radius': 12,
                    'circle-color': riskColor,
                    'circle-opacity': 0.3,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#fff'
                }
            });
        } else {
            appState.map.getSource('scan-highlight').setData({
                "type": "Feature",
                "geometry": { "type": "Point", "coordinates": [lngLat.lng, lngLat.lat] }
            });
            appState.map.setPaintProperty('scan-highlight-layer', 'circle-color', riskColor);
            appState.map.setLayoutProperty('scan-highlight-layer', 'visibility', 'visible');
        }
    }
}

/**
 * Calculate population at risk for a given tile location
 * Uses the population heatmap data from the API
 */
function calculatePopulationAtRisk(lngLat, riskLevel, depth) {
    const result = { count: 0, density: 'LOW' };
    
    // Get population data from API response
    const popData = appState.apiData?.populationHeatmap;
    if (!popData || !popData.features) {
        // Fallback estimation based on location and risk
        const riskMultiplier = { 'extreme': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1, 'safe': 0 };
        const basePopPerTile = 150; // Average population per tile
        const multiplier = riskMultiplier[riskLevel.toLowerCase()] || 0;
        result.count = Math.round(basePopPerTile * multiplier * (1 + depth / 100));
        result.density = multiplier > 0.6 ? 'HIGH' : (multiplier > 0.3 ? 'MEDIUM' : 'LOW');
        return result;
    }
    
    // Define tile bounds (approximate 500m x 500m tile)
    const tileSize = 0.005; // ~500m in degrees
    const minLng = lngLat.lng - tileSize / 2;
    const maxLng = lngLat.lng + tileSize / 2;
    const minLat = lngLat.lat - tileSize / 2;
    const maxLat = lngLat.lat + tileSize / 2;
    
    // Count population points within tile and sum intensities
    let totalIntensity = 0;
    let pointsInTile = 0;
    
    popData.features.forEach(feature => {
        const coords = feature.geometry.coordinates;
        if (coords[0] >= minLng && coords[0] <= maxLng && 
            coords[1] >= minLat && coords[1] <= maxLat) {
            pointsInTile++;
            totalIntensity += feature.properties.intensity || 0.5;
        }
    });
    
    // Calculate estimated population based on village total and point density
    const metadata = popData.metadata || {};
    const totalPop = metadata.estimated_population || 15000;
    const totalPoints = popData.features.length || 600;
    const popPerPoint = totalPop / totalPoints;
    
    // Population at risk = points in tile * pop per point * intensity * risk factor
    const riskFactor = { 'extreme': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.2, 'safe': 0 };
    const factor = riskFactor[riskLevel.toLowerCase()] || 0;
    
    if (pointsInTile > 0) {
        const avgIntensity = totalIntensity / pointsInTile;
        result.count = Math.round(pointsInTile * popPerPoint * avgIntensity * factor);
        result.density = avgIntensity > 0.7 ? 'HIGH' : (avgIntensity > 0.4 ? 'MEDIUM' : 'LOW');
    }
    
    return result;
}

// Initialize when DOM is ready
window.addEventListener('DOMContentLoaded', run);
