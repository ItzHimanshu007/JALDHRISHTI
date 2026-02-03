"""
Jal Drishti - Valley Data Generator
Synthetic data generation following real-world physics for a "Valley Village" topology.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass, asdict
import math

# Perlin noise implementation (simplified)
def perlin_noise_2d(shape: Tuple[int, int], scale: float = 10.0, octaves: int = 4) -> np.ndarray:
    """Generate 2D Perlin-like noise using numpy."""
    def interpolate(a0, a1, w):
        return (a1 - a0) * ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w) + a0
    
    def generate_gradient(shape):
        angles = np.random.rand(*shape) * 2 * np.pi
        return np.cos(angles), np.sin(angles)
    
    result = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        # Generate random gradients
        grid_shape = (int(shape[0] / scale * frequency) + 2, int(shape[1] / scale * frequency) + 2)
        gx, gy = generate_gradient(grid_shape)
        
        # Create coordinate grids
        x = np.linspace(0, grid_shape[0] - 1, shape[0])
        y = np.linspace(0, grid_shape[1] - 1, shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Get integer and fractional parts
        x0 = X.astype(int)
        y0 = Y.astype(int)
        x1 = np.clip(x0 + 1, 0, grid_shape[0] - 1)
        y1 = np.clip(y0 + 1, 0, grid_shape[1] - 1)
        
        # Fractional parts
        sx = X - x0
        sy = Y - y0
        
        # Dot products
        def dot_grid_gradient(ix, iy, x, y):
            dx = x - ix
            dy = y - iy
            ix = np.clip(ix, 0, grid_shape[0] - 1)
            iy = np.clip(iy, 0, grid_shape[1] - 1)
            return dx * gx[ix, iy] + dy * gy[ix, iy]
        
        n00 = dot_grid_gradient(x0, y0, X, Y)
        n10 = dot_grid_gradient(x1, y0, X, Y)
        n01 = dot_grid_gradient(x0, y1, X, Y)
        n11 = dot_grid_gradient(x1, y1, X, Y)
        
        # Interpolate
        ix0 = interpolate(n00, n10, sx)
        ix1 = interpolate(n01, n11, sx)
        value = interpolate(ix0, ix1, sy)
        
        result += value * amplitude
        max_value += amplitude
        amplitude *= 0.5
        frequency *= 2
    
    return result / max_value


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    grid_size: int = 100
    min_elevation: float = 650.0  # Riverbed elevation (meters)
    max_elevation: float = 850.0  # Ridge elevation (meters)
    center_lat: float = 11.555    # Wayanad-like coordinates
    center_lon: float = 76.135
    grid_resolution: float = 0.001  # ~100m per cell in degrees


class ValleyTerrainGenerator:
    """Generates a parabolic valley terrain heightmap."""
    
    def __init__(self, config: TerrainConfig = None):
        self.config = config or TerrainConfig()
        self.heightmap = None
        self.slope_map = None
        
    def generate_heightmap(self) -> np.ndarray:
        """
        Generate parabolic valley terrain.
        High elevation on edges, low in center (riverbed).
        """
        size = self.config.grid_size
        
        # Create parabolic base (valley shape)
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Parabolic valley - high on left/right edges, low in center
        # Using X^2 creates the valley running north-south
        parabolic = X ** 2
        
        # Add slight variation along Y axis (river gradient)
        river_gradient = 0.1 * (1 - Y)  # Higher in north, lower in south
        
        # Combine for base terrain
        base_terrain = parabolic + river_gradient
        
        # Normalize to 0-1
        base_terrain = (base_terrain - base_terrain.min()) / (base_terrain.max() - base_terrain.min())
        
        # Add Perlin noise for organic appearance
        noise = perlin_noise_2d((size, size), scale=15.0, octaves=4)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Blend: 85% parabolic structure, 15% noise
        combined = 0.85 * base_terrain + 0.15 * noise
        
        # Scale to elevation range
        elevation_range = self.config.max_elevation - self.config.min_elevation
        self.heightmap = combined * elevation_range + self.config.min_elevation
        
        return self.heightmap
    
    def calculate_slope(self) -> np.ndarray:
        """Calculate slope in degrees from heightmap."""
        if self.heightmap is None:
            self.generate_heightmap()
        
        # Calculate gradients
        dy, dx = np.gradient(self.heightmap)
        
        # Convert to slope angle (degrees)
        # Assuming each cell is ~100m
        cell_size = 100  # meters
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / cell_size)
        self.slope_map = np.degrees(slope_rad)
        
        return self.slope_map
    
    def to_geojson_points(self) -> Dict:
        """Convert heightmap to GeoJSON points with elevation."""
        if self.heightmap is None:
            self.generate_heightmap()
        
        features = []
        size = self.config.grid_size
        
        for i in range(0, size, 5):  # Sample every 5th point to reduce size
            for j in range(0, size, 5):
                lat = self.config.center_lat + (i - size/2) * self.config.grid_resolution
                lon = self.config.center_lon + (j - size/2) * self.config.grid_resolution
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "elevation": float(self.heightmap[i, j]),
                        "grid_i": i,
                        "grid_j": j
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


class VillageBoundaryGenerator:
    """Generates village boundary polygon encompassing the entire simulation area."""
    
    # Village-specific bounding boxes matching frontend SIMULATION_CONFIG
    # Format: [minLon, minLat, maxLon, maxLat]
    VILLAGE_BBOX = {
        "wayanad_meppadi": {
            "bbox": [76.10, 11.52, 76.17, 11.59],
            "name": "Wayanad Meppadi Village",
            "area_km2": 14.8
        },
        "darbhanga": {
            "bbox": [85.85, 26.12, 85.93, 26.19],
            "name": "Darbhanga District",
            "area_km2": 15.2
        },
        "dhemaji": {
            "bbox": [94.53, 27.45, 94.60, 27.51],
            "name": "Dhemaji District",
            "area_km2": 13.6
        }
    }
    
    def __init__(self, terrain: ValleyTerrainGenerator, village_id: str = "wayanad_meppadi"):
        self.terrain = terrain
        self.village_id = village_id
        
    def generate_boundary(self, elevation_threshold: float = 750.0) -> Dict:
        """
        Generate boundary polygon covering the entire simulation/flood tile area.
        Creates a natural-looking boundary with slight organic variations.
        """
        # Get village-specific bbox or default to wayanad
        village_config = self.VILLAGE_BBOX.get(self.village_id, self.VILLAGE_BBOX["wayanad_meppadi"])
        bbox = village_config["bbox"]
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Add slight padding to ensure complete coverage (2% padding)
        padding_lon = (max_lon - min_lon) * 0.02
        padding_lat = (max_lat - min_lat) * 0.02
        
        min_lon -= padding_lon
        min_lat -= padding_lat
        max_lon += padding_lon
        max_lat += padding_lat
        
        # Create boundary with organic variations for natural appearance
        # Using 48 points for smooth curves
        boundary_points = []
        num_points = 48
        
        # Generate points along the perimeter with slight organic variations
        for i in range(num_points):
            t = i / num_points
            
            # Add small noise for organic feel (±1% variation)
            noise_factor = 0.01
            noise = math.sin(t * 12 * math.pi) * noise_factor
            
            if t < 0.25:
                # Bottom edge (west to east)
                progress = t / 0.25
                lon = min_lon + progress * (max_lon - min_lon)
                lat = min_lat + noise * (max_lat - min_lat)
            elif t < 0.5:
                # Right edge (south to north)
                progress = (t - 0.25) / 0.25
                lon = max_lon + noise * (max_lon - min_lon)
                lat = min_lat + progress * (max_lat - min_lat)
            elif t < 0.75:
                # Top edge (east to west)
                progress = (t - 0.5) / 0.25
                lon = max_lon - progress * (max_lon - min_lon)
                lat = max_lat + noise * (max_lat - min_lat)
            else:
                # Left edge (north to south)
                progress = (t - 0.75) / 0.25
                lon = min_lon + noise * (max_lon - min_lon)
                lat = max_lat - progress * (max_lat - min_lat)
            
            boundary_points.append([lon, lat])
        
        # Close the polygon
        boundary_points.append(boundary_points[0])
        
        return {
            "type": "Feature",
            "properties": {
                "name": village_config["name"],
                "village_id": self.village_id,
                "type": "administrative",
                "habitable_area_km2": village_config["area_km2"],
                "bbox": bbox
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [boundary_points]
            }
        }


class InfrastructureGenerator:
    """Smart placement of infrastructure based on terrain."""
    
    def __init__(self, terrain: ValleyTerrainGenerator):
        self.terrain = terrain
        
    def generate_pois(self) -> Dict:
        """
        Generate Points of Interest with smart placement.
        - High ground: Hospital, Police Station (safe)
        - Low ground: Schools, Market (at risk)
        """
        if self.terrain.heightmap is None:
            self.terrain.generate_heightmap()
        
        config = self.terrain.config
        size = config.grid_size
        heightmap = self.terrain.heightmap
        
        features = []
        
        # Find high ground locations (upper 20% of elevation)
        high_threshold = np.percentile(heightmap, 80)
        # Find low ground locations (lower 30% of elevation)
        low_threshold = np.percentile(heightmap, 30)
        
        def add_poi(name: str, poi_type: str, is_safe: bool, quadrant: str):
            """Add a POI in the specified quadrant."""
            # Determine search area based on quadrant
            if quadrant == "NW":
                i_range, j_range = (0, size//2), (0, size//2)
            elif quadrant == "NE":
                i_range, j_range = (0, size//2), (size//2, size)
            elif quadrant == "SW":
                i_range, j_range = (size//2, size), (0, size//2)
            elif quadrant == "SE":
                i_range, j_range = (size//2, size), (size//2, size)
            else:  # Center
                i_range, j_range = (size//3, 2*size//3), (size//3, 2*size//3)
            
            # Find suitable location
            sub_heightmap = heightmap[i_range[0]:i_range[1], j_range[0]:j_range[1]]
            
            if is_safe:
                # Find highest point in region
                idx = np.unravel_index(np.argmax(sub_heightmap), sub_heightmap.shape)
            else:
                # Find lowest point in region
                idx = np.unravel_index(np.argmin(sub_heightmap), sub_heightmap.shape)
            
            i = i_range[0] + idx[0]
            j = j_range[0] + idx[1]
            
            lat = config.center_lat + (i - size/2) * config.grid_resolution
            lon = config.center_lon + (j - size/2) * config.grid_resolution
            elevation = float(heightmap[i, j])
            
            return {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "type": poi_type,
                    "is_safe_haven": is_safe,
                    "elevation_m": elevation,
                    "risk_level": "low" if is_safe else "high"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            }
        
        # Safe locations (high ground)
        features.append(add_poi("District Hospital", "hospital", True, "NW"))
        features.append(add_poi("Police Station", "police", True, "NE"))
        
        # At-risk locations (low ground / riverbed)
        features.append(add_poi("Government School", "school", False, "center"))
        features.append(add_poi("Primary School", "school", False, "SW"))
        features.append(add_poi("Main Market", "market", False, "center"))
        
        # Additional infrastructure
        features.append(add_poi("Fire Station", "fire_station", True, "NW"))
        features.append(add_poi("Community Hall", "shelter", False, "SE"))
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


class PopulationHeatmapGenerator:
    """Generates population distribution heatmap."""
    
    def __init__(self, terrain: ValleyTerrainGenerator, num_points: int = 500):
        self.terrain = terrain
        self.num_points = num_points
        
    def generate_heatmap(self) -> Dict:
        """
        Generate population heatmap with Gaussian distribution.
        Higher density near riverbed (where people typically settle).
        """
        if self.terrain.heightmap is None:
            self.terrain.generate_heightmap()
        
        config = self.terrain.config
        size = config.grid_size
        heightmap = self.terrain.heightmap
        
        features = []
        
        # Riverbed center (lowest elevation area)
        # More people settle near the river
        center_i, center_j = size // 2, size // 2
        
        # Generate points with Gaussian distribution
        np.random.seed(42)  # Reproducibility
        
        for _ in range(self.num_points):
            # Gaussian distribution centered on riverbed
            # Sigma controls spread - smaller = more concentrated
            i = int(np.clip(np.random.normal(center_i, size * 0.25), 0, size - 1))
            j = int(np.clip(np.random.normal(center_j, size * 0.15), 0, size - 1))  # Narrower E-W
            
            # Convert to coordinates
            lat = config.center_lat + (i - size/2) * config.grid_resolution
            lon = config.center_lon + (j - size/2) * config.grid_resolution
            
            # Intensity based on elevation (lower = more people)
            elevation = heightmap[i, j]
            min_elev = heightmap.min()
            max_elev = heightmap.max()
            
            # Inverse relationship: lower elevation = higher intensity
            intensity = 1.0 - (elevation - min_elev) / (max_elev - min_elev)
            intensity = intensity * 0.8 + 0.2  # Scale to 0.2-1.0 range
            
            # Add some randomness
            intensity = min(1.0, max(0.1, intensity + np.random.uniform(-0.1, 0.1)))
            
            features.append({
                "type": "Feature",
                "properties": {
                    "intensity": float(intensity),
                    "weight": float(intensity),
                    "elevation_m": float(elevation)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


def generate_all_data(config: TerrainConfig = None, village_id: str = "wayanad_meppadi") -> Dict[str, Any]:
    """Generate all synthetic data for the specified village."""
    if config is None:
        config = TerrainConfig()
    
    # Initialize generators
    terrain = ValleyTerrainGenerator(config)
    terrain.generate_heightmap()
    terrain.calculate_slope()
    
    boundary_gen = VillageBoundaryGenerator(terrain, village_id=village_id)
    infra_gen = InfrastructureGenerator(terrain)
    pop_gen = PopulationHeatmapGenerator(terrain)
    
    return {
        "terrain": {
            "heightmap": terrain.heightmap.tolist(),
            "slope_map": terrain.slope_map.tolist(),
            "config": asdict(config)
        },
        "boundary": boundary_gen.generate_boundary(),
        "infrastructure": infra_gen.generate_pois(),
        "population_heatmap": pop_gen.generate_heatmap(),
        "terrain_points": terrain.to_geojson_points()
    }


if __name__ == "__main__":
    # Test generation
    data = generate_all_data()
    print(f"Generated {len(data['population_heatmap']['features'])} population points")
    print(f"Generated {len(data['infrastructure']['features'])} POIs")
    print(f"Boundary has {len(data['boundary']['geometry']['coordinates'][0])} vertices")
