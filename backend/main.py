"""
Jal Drishti - FastAPI Backend
Main API server for flood simulation dashboard.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json

# Local imports
from .generators import (
    ValleyTerrainGenerator, 
    TerrainConfig, 
    VillageBoundaryGenerator,
    InfrastructureGenerator,
    PopulationHeatmapGenerator,
    generate_all_data
)
from .simulation import FloodSimulator, run_simulation
from .routing import RescueRouter, RouteConfig, find_rescue_path

# Initialize FastAPI app
app = FastAPI(
    title="Jal Drishti API",
    description="Flood Simulation & Rescue Mission Control Backend",
    version="2.0.0"
)


# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Global state for caching - now village-specific
_cache: Dict[str, Any] = {
    "current_village": None,
    "terrain": None,
    "boundary": None,
    "infrastructure": None,
    "population": None,
    "heightmap": None,
    "last_init": None
}


def get_terrain_config(village_id: str = "wayanad_meppadi") -> TerrainConfig:
    """Get terrain configuration for a village."""
    configs = {
        "wayanad_meppadi": TerrainConfig(
            center_lat=11.555,
            center_lon=76.135,
            min_elevation=650.0,
            max_elevation=850.0
        ),
        "darbhanga": TerrainConfig(
            center_lat=26.120,
            center_lon=85.900,
            min_elevation=45.0,
            max_elevation=55.0  # Flat terrain
        ),
        "dhemaji": TerrainConfig(
            center_lat=27.480,
            center_lon=94.560,
            min_elevation=100.0,
            max_elevation=150.0
        )
    }
    return configs.get(village_id, configs["wayanad_meppadi"])


def initialize_data(village_id: str = "wayanad_meppadi", force: bool = False):
    """Initialize or refresh cached data for a specific village."""
    # Force regeneration if village changed or force flag is set
    if not force and _cache.get("current_village") == village_id and _cache["heightmap"] is not None:
        return
    
    config = get_terrain_config(village_id)
    
    # Generate terrain
    terrain = ValleyTerrainGenerator(config, village_id=village_id)
    heightmap = terrain.generate_heightmap()
    terrain.calculate_slope()
    
    # Generate other data
    boundary_gen = VillageBoundaryGenerator(terrain, village_id=village_id)
    infra_gen = InfrastructureGenerator(terrain)
    pop_gen = PopulationHeatmapGenerator(terrain, village_id=village_id)
    
    # Cache everything with current village
    _cache["current_village"] = village_id
    _cache["heightmap"] = heightmap
    _cache["terrain"] = terrain
    _cache["boundary"] = boundary_gen.generate_boundary()
    _cache["infrastructure"] = infra_gen.generate_pois()
    _cache["population"] = pop_gen.generate_heatmap()
    _cache["config"] = config
    _cache["last_init"] = datetime.now()


# ============================================
# API Endpoints
# ============================================

@app.get("/api/health")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Jal Drishti API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/weather")
async def get_weather(
    lat: float = Query(11.555, description="Latitude"),
    lon: float = Query(76.135, description="Longitude"),
    days: int = Query(7, description="Forecast days (1-7)")
):
    """
    Fetch real-time weather data from Open-Meteo API.
    Returns current conditions and 7-day precipitation forecast.
    """
    try:
        # Validate days
        days = min(max(days, 1), 7)
        
        # Open-Meteo API call
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m,precipitation"
            f"&daily=precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min"
            f"&timezone=auto"
            f"&forecast_days={days}"
        )
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
        
        # Format response for frontend
        current = data.get("current_weather", {})
        daily = data.get("daily", {})
        
        # Build 7-day forecast
        forecast = []
        dates = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        prob = daily.get("precipitation_probability_max", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        
        for i in range(min(len(dates), days)):
            forecast.append({
                "date": dates[i],
                "day_name": datetime.fromisoformat(dates[i]).strftime("%a"),
                "precipitation_mm": precip[i] if i < len(precip) else 0,
                "probability_percent": prob[i] if i < len(prob) else 0,
                "temp_max_c": temp_max[i] if i < len(temp_max) else None,
                "temp_min_c": temp_min[i] if i < len(temp_min) else None
            })
        
        return {
            "status": "success",
            "location": {"lat": lat, "lon": lon},
            "current": {
                "temperature_c": current.get("temperature"),
                "windspeed_kmh": current.get("windspeed"),
                "weather_code": current.get("weathercode"),
                "is_day": current.get("is_day", 1) == 1
            },
            "forecast": forecast,
            "total_precipitation_mm": sum(precip[:days]) if precip else 0,
            "fetched_at": datetime.now().isoformat()
        }
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Weather service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather: {str(e)}")


@app.get("/api/init")
async def init_dashboard(
    village_id: str = Query("wayanad_meppadi", description="Village identifier")
):
    """
    Initialize dashboard with all required data.
    Returns map config, village boundary, POIs, and population heatmap.
    """
    try:
        # Initialize data for village
        initialize_data(village_id)
        
        config = _cache["config"]
        
        return {
            "status": "success",
            "village_id": village_id,
            "map_config": {
                "center": [config.center_lon, config.center_lat],
                "zoom": 13,
                "pitch": 60,
                "bearing": -10,
                "satellite_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "terrain_url": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
            },
            "boundary": _cache["boundary"],
            "infrastructure": _cache["infrastructure"],
            "population_heatmap": _cache["population"],
            "terrain_config": {
                "grid_size": config.grid_size,
                "elevation_range": [config.min_elevation, config.max_elevation],
                "center": [config.center_lat, config.center_lon]
            },
            "initialized_at": _cache["last_init"].isoformat() if _cache["last_init"] else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.get("/api/simulate")
async def simulate_flood(
    rainfall: float = Query(100.0, ge=0, le=500, description="Rainfall in mm"),
    village_id: str = Query("wayanad_meppadi", description="Village identifier"),
    format: str = Query("polygons", description="Output format: polygons or grid")
):
    """
    Run flood simulation for given rainfall amount.
    Returns GeoJSON with 3 time-step flood polygons (T1, T2, T3).
    """
    try:
        # Ensure data is initialized
        initialize_data(village_id)
        
        if _cache["heightmap"] is None:
            raise HTTPException(status_code=500, detail="Terrain not initialized")
        
        config = _cache["config"]
        
        # Run simulation
        result = run_simulation(
            heightmap=_cache["heightmap"],
            rainfall_mm=rainfall,
            center_lat=config.center_lat,
            center_lon=config.center_lon,
            output_format=format
        )
        
        # Add metadata
        return {
            "status": "success",
            "rainfall_mm": rainfall,
            "village_id": village_id,
            "simulation": result,
            "summary": {
                "timesteps": len(result.get("features", [])),
                "max_severity": max(
                    (f["properties"].get("severity", "low") for f in result.get("features", [])),
                    default="none"
                )
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.get("/api/rescue")
async def find_rescue_route(
    lat: float = Query(..., description="User latitude"),
    lon: float = Query(..., description="User longitude"),
    village_id: str = Query("wayanad_meppadi", description="Village identifier"),
    flood_level: float = Query(0.0, ge=0, le=5, description="Current flood depth factor")
):
    """
    Find safest rescue path from user location to nearest safe haven.
    Uses A* pathfinding considering terrain and flood conditions.
    """
    try:
        # Ensure data is initialized
        initialize_data(village_id)
        
        if _cache["heightmap"] is None or _cache["infrastructure"] is None:
            raise HTTPException(status_code=500, detail="Data not initialized")
        
        config = _cache["config"]
        
        # Generate flood depth if specified
        flood_depth = None
        if flood_level > 0:
            simulator = FloodSimulator(_cache["heightmap"])
            depths = simulator.simulate(flood_level * 100)  # Convert to mm equivalent
            flood_depth = depths.get("t2", None)  # Use middle time step
        
        # Create route config
        route_config = RouteConfig(
            grid_size=config.grid_size,
            center_lat=config.center_lat,
            center_lon=config.center_lon,
            grid_resolution=config.grid_resolution
        )
        
        # Find rescue path
        result = find_rescue_path(
            user_lat=lat,
            user_lon=lon,
            heightmap=_cache["heightmap"],
            safe_havens=_cache["infrastructure"]["features"],
            flood_depth=flood_depth,
            config=route_config
        )
        
        return {
            "status": result.get("status", "unknown"),
            "rescue_path": result,
            "user_location": {"lat": lat, "lon": lon},
            "flood_level": flood_level,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


@app.get("/api/terrain")
async def get_terrain(
    village_id: str = Query("wayanad_meppadi", description="Village identifier"),
    format: str = Query("summary", description="Output format: summary, points, or raw")
):
    """
    Get terrain data for visualization.
    """
    try:
        initialize_data(village_id)
        
        if _cache["terrain"] is None:
            raise HTTPException(status_code=500, detail="Terrain not initialized")
        
        terrain = _cache["terrain"]
        heightmap = _cache["heightmap"]
        
        if format == "raw":
            return {
                "heightmap": heightmap.tolist(),
                "slope_map": terrain.slope_map.tolist() if terrain.slope_map is not None else None
            }
        elif format == "points":
            return terrain.to_geojson_points()
        else:
            return {
                "elevation_min": float(heightmap.min()),
                "elevation_max": float(heightmap.max()),
                "elevation_mean": float(heightmap.mean()),
                "grid_size": terrain.config.grid_size,
                "center": [terrain.config.center_lat, terrain.config.center_lon]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving terrain: {str(e)}")


@app.post("/api/reset")
async def reset_cache(
    village_id: str = Query("wayanad_meppadi", description="Village identifier")
):
    """Reset cached data and regenerate."""
    try:
        initialize_data(village_id, force=True)
        return {
            "status": "success",
            "message": f"Cache reset for {village_id}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Mount static files
# Get absolute path to frontend directory
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {frontend_path}")


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    # Use string reference for reload to work
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
