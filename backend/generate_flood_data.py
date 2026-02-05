"""
Jal Drishti - Realistic Flood Data Generator
Pre-calculates flood simulations for all villages and various rainfall levels.
"""

import os
import json
import numpy as np
from generators import ValleyTerrainGenerator, TerrainConfig, VillageBoundaryGenerator
from simulation import FloodSimulator, SimulationConfig

def get_village_configs():
    return {
        "wayanad_meppadi": {
            "center": [11.555, 76.135],
            "elev": [650.0, 850.0],
            "rainfall": [100, 200, 300]
        },
        "darbhanga": {
            "center": [26.120, 85.900],
            "elev": [45.0, 55.0],
            "rainfall": [150, 300, 450]
        },
        "dhemaji": {
            "center": [27.480, 94.560],
            "elev": [100.0, 150.0],
            "rainfall": [200, 400, 600]
        }
    }

def generate_static_data():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard", "data", "simulations")
    os.makedirs(base_dir, exist_ok=True)
    
    configs = get_village_configs()
    
    for vid, cfg in configs.items():
        print(f"Generating data for {vid}...")
        
        t_cfg = TerrainConfig(
            center_lat=cfg["center"][0],
            center_lon=cfg["center"][1],
            min_elevation=cfg["elev"][0],
            max_elevation=cfg["elev"][1]
        )
        
        terrain = ValleyTerrainGenerator(t_cfg, village_id=vid)
        heightmap = terrain.generate_heightmap()
        
        simulator = FloodSimulator(heightmap)
        
        v_dir = os.path.join(base_dir, vid)
        os.makedirs(v_dir, exist_ok=True)
        
        for rf in cfg["rainfall"]:
            print(f"  Simulation for {rf}mm rainfall...")
            simulator.simulate(rf)
            
            # Use high-detail grid output for realism
            geojson = simulator.to_geojson_grid(
                center_lat=cfg["center"][0],
                center_lon=cfg["center"][1],
                grid_resolution=t_cfg.grid_resolution
            )
            
            filename = f"flood_{rf}mm.geojson"
            with open(os.path.join(v_dir, filename), "w") as f:
                json.dump(geojson, f)
                
    print("Done!")

if __name__ == "__main__":
    generate_static_data()
