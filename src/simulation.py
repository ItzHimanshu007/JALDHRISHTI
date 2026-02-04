"""
Jal Drishti - Flood Simulation Engine
Physics-based flood simulation using D8 flow direction and water accumulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class SimulationConfig:
    """Configuration for flood simulation."""
    grid_size: int = 100
    time_steps: int = 3  # T1, T2, T3
    infiltration_rate: float = 0.3  # Fraction of water that infiltrates
    runoff_coefficient: float = 0.7  # Fraction that becomes runoff
    manning_n: float = 0.035  # Manning's roughness coefficient
    cell_size_m: float = 100.0  # Cell size in meters


class D8FlowDirector:
    """
    D8 Flow Direction Algorithm.
    Water flows to the lowest of 8 neighbors.
    """
    
    # Direction codes and offsets
    # 32 64 128
    # 16  X   1
    #  8  4   2
    DIRECTIONS = {
        1: (0, 1),    # E
        2: (1, 1),    # SE
        4: (1, 0),    # S
        8: (1, -1),   # SW
        16: (0, -1),  # W
        32: (-1, -1), # NW
        64: (-1, 0),  # N
        128: (-1, 1)  # NE
    }
    
    def __init__(self, heightmap: np.ndarray):
        self.heightmap = heightmap
        self.flow_direction = None
        self.flow_accumulation = None
        
    def calculate_flow_direction(self) -> np.ndarray:
        """Calculate D8 flow direction for each cell."""
        rows, cols = self.heightmap.shape
        self.flow_direction = np.zeros((rows, cols), dtype=np.int32)
        
        for i in range(rows):
            for j in range(cols):
                min_drop = 0
                direction = 0
                
                for code, (di, dj) in self.DIRECTIONS.items():
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < rows and 0 <= nj < cols:
                        drop = self.heightmap[i, j] - self.heightmap[ni, nj]
                        
                        # Diagonal cells are sqrt(2) further
                        if abs(di) + abs(dj) == 2:
                            drop /= 1.414
                        
                        if drop > min_drop:
                            min_drop = drop
                            direction = code
                
                self.flow_direction[i, j] = direction
        
        return self.flow_direction
    
    def calculate_flow_accumulation(self) -> np.ndarray:
        """Calculate flow accumulation (upstream contributing area)."""
        if self.flow_direction is None:
            self.calculate_flow_direction()
        
        rows, cols = self.heightmap.shape
        self.flow_accumulation = np.ones((rows, cols), dtype=np.float64)
        
        # Sort cells by elevation (highest first)
        flat_indices = np.argsort(self.heightmap.flatten())[::-1]
        
        for idx in flat_indices:
            i, j = np.unravel_index(idx, (rows, cols))
            direction = self.flow_direction[i, j]
            
            if direction in self.DIRECTIONS:
                di, dj = self.DIRECTIONS[direction]
                ni, nj = i + di, j + dj
                
                if 0 <= ni < rows and 0 <= nj < cols:
                    self.flow_accumulation[ni, nj] += self.flow_accumulation[i, j]
        
        return self.flow_accumulation


class FloodSimulator:
    """
    Water Balance Flood Simulation.
    Simulates water entering, flowing, and accumulating in the terrain.
    """
    
    def __init__(self, heightmap: np.ndarray, config: SimulationConfig = None):
        self.heightmap = heightmap
        self.config = config or SimulationConfig()
        self.flow_director = D8FlowDirector(heightmap)
        self.water_depth = None
        
    def simulate(self, rainfall_mm: float) -> Dict[str, np.ndarray]:
        """
        Run flood simulation with diffusion and micro-noise.
        """
        # Calculate flow characteristics
        flow_dir = self.flow_director.calculate_flow_direction()
        flow_acc = self.flow_director.calculate_flow_accumulation()
        
        rows, cols = self.heightmap.shape
        cell_area_m2 = self.config.cell_size_m ** 2
        
        # Convert rainfall to water volume per cell
        rainfall_m = rainfall_mm / 1000.0
        
        # Initialize results
        results = {}
        time_factors = {"t1": 0.3, "t2": 0.6, "t3": 1.0}
        
        # Normalize flow accumulation
        max_flow_acc = flow_acc.max() if flow_acc.max() > 0 else 1.0
        normalized_acc = flow_acc / max_flow_acc
        
        # Riverbed weighting
        riverbed_threshold = np.percentile(self.heightmap, 15)
        is_riverbed = self.heightmap < riverbed_threshold
        
        # Terrain normalization for gravity effect
        elev_min, elev_max = self.heightmap.min(), self.heightmap.max()
        elev_normalized = (self.heightmap - elev_min) / (elev_max - elev_min + 1e-6)
        elevation_factor = 1.0 - elev_normalized
        
        # Micro-terrain noise for edge realism
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (rows, cols))
        
        current_water = np.zeros((rows, cols))
        
        for timestep, factor in time_factors.items():
            # 1. Base accumulation from rainfall and flow
            # Higher accumulation in riverbeds and low points
            base_surge = normalized_acc * rainfall_m * factor * 8 
            river_surge = np.where(is_riverbed, rainfall_m * factor * 3, 0)
            
            # Combine and add noise
            new_depth = (base_surge + river_surge) * (0.8 + elevation_factor * 1.2)
            new_depth += noise * rainfall_m
            
            # 2. Diffusion step (Simulate lateral flattening)
            # Simple 4-neighbor smoothing to mimic gravity leveling
            for _ in range(3): # 3 iterations of diffusion
                smoothed = new_depth.copy()
                # North
                smoothed[1:, :] += new_depth[:-1, :] * 0.1
                # South 
                smoothed[:-1, :] += new_depth[1:, :] * 0.1
                # East
                smoothed[:, 1:] += new_depth[:, :-1] * 0.1
                # West
                smoothed[:, :-1] += new_depth[:, 1:] * 0.1
                new_depth = (smoothed / 1.4) # Normalize
            
            # Clip and store
            water_depth = np.clip(new_depth, 0, 5.0)
            results[timestep] = water_depth
        
        self.water_depth = results
        return results
    
    def get_flooded_areas(self, depth_threshold: float = 0.1) -> Dict[str, np.ndarray]:
        """Get binary masks of flooded areas for each time step."""
        if self.water_depth is None:
            raise ValueError("Run simulate() first")
        
        return {
            ts: (depths > depth_threshold).astype(np.int32)
            for ts, depths in self.water_depth.items()
        }
    
    def to_geojson_polygons(self, center_lat: float, center_lon: float, 
                           grid_resolution: float = 0.001) -> Dict:
        """
        Convert flood simulation results to GeoJSON polygons.
        Each time step becomes a separate feature.
        """
        if self.water_depth is None:
            raise ValueError("Run simulate() first")
        
        features = []
        rows, cols = self.heightmap.shape
        
        for timestep, depth_map in self.water_depth.items():
            # Create flood polygon by finding cells with significant water
            flooded = depth_map > 0.1
            
            if not flooded.any():
                continue
            
            # Create simplified polygon from flooded cells
            # Find convex hull of flooded area for simplicity
            flooded_coords = np.argwhere(flooded)
            
            if len(flooded_coords) < 3:
                continue
            
            # Convert to lat/lon coordinates
            latlons = []
            for i, j in flooded_coords:
                lat = center_lat + (i - rows/2) * grid_resolution
                lon = center_lon + (j - cols/2) * grid_resolution
                latlons.append([lon, lat])
            
            # Create convex hull (simplified boundary)
            from scipy.spatial import ConvexHull
            try:
                points = np.array(latlons)
                hull = ConvexHull(points)
                hull_points = points[hull.vertices].tolist()
                hull_points.append(hull_points[0])  # Close polygon
            except Exception:
                # Fallback: use bounding box
                lons = [p[0] for p in latlons]
                lats = [p[1] for p in latlons]
                hull_points = [
                    [min(lons), min(lats)],
                    [max(lons), min(lats)],
                    [max(lons), max(lats)],
                    [min(lons), max(lats)],
                    [min(lons), min(lats)]
                ]
            
            # Calculate statistics
            max_depth = float(depth_map.max())
            avg_depth = float(depth_map[flooded].mean())
            area_km2 = float(flooded.sum() * (grid_resolution * 111) ** 2)  # Approx km²
            
            features.append({
                "type": "Feature",
                "properties": {
                    "timestep": timestep,
                    "max_depth_m": round(max_depth, 2),
                    "avg_depth_m": round(avg_depth, 2),
                    "flooded_area_km2": round(area_km2, 3),
                    "severity": "extreme" if max_depth > 2 else "high" if max_depth > 1 else "moderate"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [hull_points]
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def to_geojson_grid(self, center_lat: float, center_lon: float,
                        grid_resolution: float = 0.001) -> Dict:
        """
        Convert flood simulation to grid cells for detailed visualization.
        Returns cells with significant water depth.
        """
        if self.water_depth is None:
            raise ValueError("Run simulate() first")
        
        all_features = []
        rows, cols = self.heightmap.shape
        
        for timestep, depth_map in self.water_depth.items():
            features = []
            
            # Sample grid for visualization (every 2nd cell to reduce data)
            for i in range(0, rows, 2):
                for j in range(0, cols, 2):
                    depth = depth_map[i, j]
                    
                    if depth > 0.05:  # Only include cells with water
                        lat = center_lat + (i - rows/2) * grid_resolution
                        lon = center_lon + (j - cols/2) * grid_resolution
                        
                        # Create small polygon for cell
                        half_res = grid_resolution * 0.5
                        cell_coords = [
                            [lon - half_res, lat - half_res],
                            [lon + half_res, lat - half_res],
                            [lon + half_res, lat + half_res],
                            [lon - half_res, lat + half_res],
                            [lon - half_res, lat - half_res]
                        ]
                        
                        # Risk value based on depth
                        if depth > 2.0:
                            risk_value = 4  # Extreme
                        elif depth > 1.0:
                            risk_value = 3  # High
                        elif depth > 0.5:
                            risk_value = 2  # Medium
                        else:
                            risk_value = 1  # Low
                        
                        features.append({
                            "type": "Feature",
                            "properties": {
                                "depth_m": round(float(depth), 2),
                                "value": risk_value,
                                "timestep": timestep
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [cell_coords]
                            }
                        })
            
            all_features.extend(features)
        
        return {
            "type": "FeatureCollection",
            "features": all_features
        }


def run_simulation(heightmap: np.ndarray, rainfall_mm: float,
                   center_lat: float = 11.555, center_lon: float = 76.135,
                   output_format: str = "polygons") -> Dict:
    """
    Main entry point for flood simulation.
    
    Args:
        heightmap: 2D numpy array of elevations
        rainfall_mm: Rainfall amount (0-300mm)
        center_lat: Center latitude for output coordinates
        center_lon: Center longitude for output coordinates
        output_format: "polygons" for simplified shapes, "grid" for detailed cells
        
    Returns:
        GeoJSON FeatureCollection with flood data
    """
    simulator = FloodSimulator(heightmap)
    simulator.simulate(rainfall_mm)
    
    if output_format == "polygons":
        return simulator.to_geojson_polygons(center_lat, center_lon)
    else:
        return simulator.to_geojson_grid(center_lat, center_lon)


if __name__ == "__main__":
    # Test with synthetic heightmap
    from generators import ValleyTerrainGenerator, TerrainConfig
    
    config = TerrainConfig()
    terrain = ValleyTerrainGenerator(config)
    heightmap = terrain.generate_heightmap()
    
    # Run simulation
    result = run_simulation(heightmap, rainfall_mm=150)
    print(f"Generated {len(result['features'])} flood polygons")
    
    for feature in result['features']:
        props = feature['properties']
        print(f"  {props['timestep']}: max depth {props['max_depth_m']}m, "
              f"area {props['flooded_area_km2']}km², severity: {props['severity']}")
