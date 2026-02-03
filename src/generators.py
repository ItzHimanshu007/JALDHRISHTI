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
    """Generates population distribution heatmap within village boundaries."""
    
    # Village-specific configurations with unique population patterns
    VILLAGE_CONFIG = {
        "wayanad_meppadi": {
            "bbox": [76.10, 11.52, 76.17, 11.59],
            "name": "Wayanad Meppadi",
            "total_population": 15000,
            "pattern": "hill_settlement",  # Scattered settlements on hillside
            "clusters": [
                # Main town center (high density)
                {"center": [76.135, 11.555], "radius": 0.015, "density": 0.95, "weight": 0.35},
                # Market area
                {"center": [76.128, 11.545], "radius": 0.012, "density": 0.85, "weight": 0.20},
                # Temple/religious area settlement
                {"center": [76.145, 11.565], "radius": 0.010, "density": 0.75, "weight": 0.15},
                # School vicinity
                {"center": [76.115, 11.535], "radius": 0.008, "density": 0.70, "weight": 0.10},
                # Agricultural workers settlements (scattered)
                {"center": [76.155, 11.575], "radius": 0.018, "density": 0.45, "weight": 0.12},
                # Tea estate worker housing
                {"center": [76.120, 11.580], "radius": 0.012, "density": 0.55, "weight": 0.08},
            ]
        },
        "darbhanga": {
            "bbox": [85.85, 26.12, 85.93, 26.19],
            "name": "Darbhanga",
            "total_population": 45000,
            "pattern": "river_plain",  # Dense settlements along river plains
            "clusters": [
                # Main city center (very high density)
                {"center": [85.89, 26.155], "radius": 0.020, "density": 0.98, "weight": 0.30},
                # Old town area
                {"center": [85.87, 26.145], "radius": 0.018, "density": 0.90, "weight": 0.20},
                # Railway station vicinity
                {"center": [85.91, 26.165], "radius": 0.015, "density": 0.85, "weight": 0.15},
                # College/educational hub
                {"center": [85.86, 26.175], "radius": 0.012, "density": 0.80, "weight": 0.12},
                # Industrial/commercial zone
                {"center": [85.92, 26.135], "radius": 0.016, "density": 0.75, "weight": 0.10},
                # Riverside settlements (flood-prone, high risk)
                {"center": [85.88, 26.125], "radius": 0.022, "density": 0.70, "weight": 0.08},
                # Agricultural village
                {"center": [85.855, 26.185], "radius": 0.010, "density": 0.50, "weight": 0.05},
            ]
        },
        "dhemaji": {
            "bbox": [94.53, 27.45, 94.60, 27.51],
            "name": "Dhemaji",
            "total_population": 12000,
            "pattern": "brahmaputra_flood_plain",  # Linear settlements along elevated areas
            "clusters": [
                # District headquarters (main town)
                {"center": [94.565, 27.48], "radius": 0.018, "density": 0.90, "weight": 0.30},
                # Hospital/administrative area
                {"center": [94.555, 27.475], "radius": 0.012, "density": 0.85, "weight": 0.18},
                # Market/bazaar area
                {"center": [94.575, 27.485], "radius": 0.010, "density": 0.82, "weight": 0.15},
                # Riverbank fishing community (high flood risk)
                {"center": [94.545, 27.455], "radius": 0.020, "density": 0.65, "weight": 0.12},
                # Northern agricultural settlement
                {"center": [94.585, 27.500], "radius": 0.015, "density": 0.55, "weight": 0.10},
                # Tribal hamlet (scattered)
                {"center": [94.540, 27.495], "radius": 0.016, "density": 0.45, "weight": 0.08},
                # Tea garden workers colony
                {"center": [94.592, 27.465], "radius": 0.008, "density": 0.60, "weight": 0.07},
            ]
        }
    }
    
    def __init__(self, terrain: ValleyTerrainGenerator, village_id: str = "wayanad_meppadi", num_points: int = 600):
        self.terrain = terrain
        self.village_id = village_id
        self.num_points = num_points
        
    def generate_heatmap(self) -> Dict:
        """
        Generate population heatmap with realistic, village-specific distribution.
        Each village has unique settlement patterns based on geography and demographics.
        """
        # Get village-specific configuration
        config = self.VILLAGE_CONFIG.get(self.village_id, self.VILLAGE_CONFIG["wayanad_meppadi"])
        bbox = config["bbox"]
        clusters = config["clusters"]
        pattern = config["pattern"]
        
        min_lon, min_lat, max_lon, max_lat = bbox
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        
        features = []
        
        # Set unique seed per village for reproducibility
        seed_map = {"wayanad_meppadi": 42, "darbhanga": 137, "dhemaji": 256}
        np.random.seed(seed_map.get(self.village_id, 42))
        
        # Calculate points per cluster based on weights
        total_weight = sum(c["weight"] for c in clusters)
        points_per_cluster = [(c, int(self.num_points * c["weight"] / total_weight)) for c in clusters]
        
        # Add remaining points to first cluster
        assigned = sum(p[1] for p in points_per_cluster)
        if assigned < self.num_points:
            points_per_cluster[0] = (points_per_cluster[0][0], points_per_cluster[0][1] + (self.num_points - assigned))
        
        for cluster, num_pts in points_per_cluster:
            cluster_center = cluster["center"]
            cluster_radius = cluster["radius"]
            base_density = cluster["density"]
            
            for _ in range(num_pts):
                # Generate point within cluster using Gaussian distribution
                lon = np.random.normal(cluster_center[0], cluster_radius * 0.5)
                lat = np.random.normal(cluster_center[1], cluster_radius * 0.5)
                
                # Ensure within bbox
                lon = np.clip(lon, min_lon, max_lon)
                lat = np.clip(lat, min_lat, max_lat)
                
                # Calculate distance from cluster center
                dist = math.sqrt(
                    ((lon - cluster_center[0]) / cluster_radius) ** 2 +
                    ((lat - cluster_center[1]) / cluster_radius) ** 2
                )
                
                # Intensity based on distance from cluster center and base density
                falloff = max(0, 1 - dist * 0.5)
                
                # Apply pattern-specific intensity modifiers
                if pattern == "hill_settlement":
                    # Wayanad: More scattered, lower overall density
                    intensity = base_density * falloff * np.random.uniform(0.6, 1.0)
                elif pattern == "river_plain":
                    # Darbhanga: Higher density, more uniform clusters
                    intensity = base_density * falloff * np.random.uniform(0.75, 1.0)
                elif pattern == "brahmaputra_flood_plain":
                    # Dhemaji: Linear patterns, moderate density
                    # Add linear bias along east-west axis
                    linear_factor = 1.0 - abs(lat - 27.48) * 5
                    linear_factor = max(0.4, min(1.0, linear_factor))
                    intensity = base_density * falloff * linear_factor * np.random.uniform(0.65, 1.0)
                else:
                    intensity = base_density * falloff * np.random.uniform(0.7, 1.0)
                
                # Clamp intensity
                intensity = min(1.0, max(0.1, intensity))
                
                # Determine density category
                if intensity > 0.75:
                    density_cat = "high"
                elif intensity > 0.45:
                    density_cat = "medium"
                else:
                    density_cat = "low"
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "intensity": float(intensity),
                        "weight": float(intensity),
                        "population_density": density_cat,
                        "cluster_type": self._get_cluster_type(cluster, pattern)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                })
        
        # Add scattered background population (10% of total)
        num_scattered = int(self.num_points * 0.1)
        for _ in range(num_scattered):
            lon = np.random.uniform(min_lon, max_lon)
            lat = np.random.uniform(min_lat, max_lat)
            intensity = np.random.uniform(0.1, 0.35)
            
            features.append({
                "type": "Feature",
                "properties": {
                    "intensity": float(intensity),
                    "weight": float(intensity),
                    "population_density": "low",
                    "cluster_type": "scattered"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "village_id": self.village_id,
                "village_name": config["name"],
                "pattern": pattern,
                "total_points": len(features),
                "estimated_population": config["total_population"]
            }
        }
    
    def _get_cluster_type(self, cluster: Dict, pattern: str) -> str:
        """Determine cluster type based on density and pattern."""
        density = cluster["density"]
        if density > 0.85:
            return "urban_center"
        elif density > 0.70:
            return "commercial"
        elif density > 0.55:
            return "residential"
        else:
            return "rural"


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
    pop_gen = PopulationHeatmapGenerator(terrain, village_id=village_id)
    
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
