"""
Jal Drishti - Rescue Routing Engine
A* pathfinding algorithm for finding safe evacuation routes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import heapq
import json


@dataclass
class RouteConfig:
    """Configuration for routing algorithm."""
    grid_size: int = 100
    cell_size_m: float = 100.0  # Cell size in meters
    center_lat: float = 11.555
    center_lon: float = 76.135
    grid_resolution: float = 0.001


@dataclass(order=True)
class PriorityNode:
    """Node for priority queue in A* algorithm."""
    f_score: float
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional[Tuple[int, int]] = field(default=None, compare=False)


class TerrainWeightCalculator:
    """Calculates traversal weights based on terrain and flood conditions."""
    
    # Weight constants
    NORMAL_WEIGHT = 1.0
    STEEP_WEIGHT = 5.0
    FLOODED_WEIGHT = 1000.0
    IMPASSABLE_WEIGHT = float('inf')
    
    # Thresholds
    STEEP_SLOPE_DEG = 15.0
    SHALLOW_FLOOD_M = 0.3
    DEEP_FLOOD_M = 1.5
    
    def __init__(self, heightmap: np.ndarray, flood_depth: Optional[np.ndarray] = None):
        self.heightmap = heightmap
        self.flood_depth = flood_depth if flood_depth is not None else np.zeros_like(heightmap)
        self.slope_map = self._calculate_slopes()
        
    def _calculate_slopes(self) -> np.ndarray:
        """Calculate slope in degrees."""
        dy, dx = np.gradient(self.heightmap)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / 100)  # Assuming 100m cells
        return np.degrees(slope_rad)
    
    def get_weight(self, i: int, j: int) -> float:
        """
        Calculate traversal weight for a cell.
        
        Returns:
            Weight value (higher = harder to traverse)
        """
        if not self._is_valid(i, j):
            return self.IMPASSABLE_WEIGHT
        
        flood = self.flood_depth[i, j]
        slope = self.slope_map[i, j]
        
        # Deep water is impassable
        if flood > self.DEEP_FLOOD_M:
            return self.IMPASSABLE_WEIGHT
        
        # Start with base weight
        weight = self.NORMAL_WEIGHT
        
        # Flooded areas are dangerous
        if flood > self.SHALLOW_FLOOD_M:
            weight = self.FLOODED_WEIGHT * (flood / self.DEEP_FLOOD_M)
        elif flood > 0:
            weight = self.NORMAL_WEIGHT + flood * 100
        
        # Steep slopes are harder
        if slope > self.STEEP_SLOPE_DEG:
            weight += self.STEEP_WEIGHT * (slope / self.STEEP_SLOPE_DEG)
        
        return weight
    
    def _is_valid(self, i: int, j: int) -> bool:
        """Check if cell is within bounds."""
        rows, cols = self.heightmap.shape
        return 0 <= i < rows and 0 <= j < cols
    
    def get_weight_grid(self) -> np.ndarray:
        """Get weight grid for visualization."""
        rows, cols = self.heightmap.shape
        weights = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                weights[i, j] = min(self.get_weight(i, j), 1000)  # Cap for viz
        
        return weights


class AStarRouter:
    """A* pathfinding algorithm for rescue routing."""
    
    # 8-directional movement (including diagonals)
    DIRECTIONS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
    ]
    
    def __init__(self, weight_calculator: TerrainWeightCalculator, config: RouteConfig = None):
        self.weights = weight_calculator
        self.config = config or RouteConfig()
        self.heightmap = weight_calculator.heightmap
        
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        for di, dj in self.DIRECTIONS:
            ni, nj = pos[0] + di, pos[1] + dj
            if self.weights._is_valid(ni, nj):
                weight = self.weights.get_weight(ni, nj)
                if weight < float('inf'):
                    neighbors.append((ni, nj))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find optimal path from start to goal using A*.
        
        Args:
            start: Starting grid position (i, j)
            goal: Goal grid position (i, j)
            
        Returns:
            List of grid positions forming the path, or None if no path exists
        """
        if not self.weights._is_valid(start[0], start[1]) or \
           not self.weights._is_valid(goal[0], goal[1]):
            return None
        
        # Check if goal is reachable
        if self.weights.get_weight(goal[0], goal[1]) >= float('inf'):
            return None
        
        # Initialize
        open_set = []
        start_node = PriorityNode(
            f_score=self._heuristic(start, goal),
            position=start,
            g_score=0
        )
        heapq.heappush(open_set, start_node)
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_scores: Dict[Tuple[int, int], float] = {start: 0}
        open_positions: Set[Tuple[int, int]] = {start}
        closed_set: Set[Tuple[int, int]] = set()
        
        iterations = 0
        max_iterations = self.config.grid_size ** 2  # Safety limit
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current_node = heapq.heappop(open_set)
            current = current_node.position
            open_positions.discard(current)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                move_cost = self.weights.get_weight(neighbor[0], neighbor[1])
                
                # Diagonal moves are longer
                di = abs(neighbor[0] - current[0])
                dj = abs(neighbor[1] - current[1])
                if di + dj == 2:
                    move_cost *= 1.414
                
                tentative_g = g_scores[current] + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    
                    if neighbor not in open_positions:
                        new_node = PriorityNode(
                            f_score=f_score,
                            position=neighbor,
                            g_score=tentative_g,
                            parent=current
                        )
                        heapq.heappush(open_set, new_node)
                        open_positions.add(neighbor)
        
        return None  # No path found
    
    def path_to_geojson(self, path: List[Tuple[int, int]]) -> Dict:
        """Convert grid path to GeoJSON LineString."""
        if not path:
            return None
        
        config = self.config
        rows = self.heightmap.shape[0]
        
        coordinates = []
        for i, j in path:
            lat = config.center_lat + (i - rows/2) * config.grid_resolution
            lon = config.center_lon + (j - rows/2) * config.grid_resolution
            coordinates.append([lon, lat])
        
        # Calculate path statistics
        total_distance = 0
        for k in range(1, len(path)):
            di = abs(path[k][0] - path[k-1][0])
            dj = abs(path[k][1] - path[k-1][1])
            if di + dj == 2:
                total_distance += config.cell_size_m * 1.414
            else:
                total_distance += config.cell_size_m
        
        return {
            "type": "Feature",
            "properties": {
                "type": "rescue_path",
                "distance_m": round(total_distance, 1),
                "distance_km": round(total_distance / 1000, 2),
                "waypoints": len(path),
                "status": "safe"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }


class RescueRouter:
    """High-level rescue routing system."""
    
    def __init__(self, heightmap: np.ndarray, safe_havens: List[Dict],
                 flood_depth: Optional[np.ndarray] = None,
                 config: RouteConfig = None):
        """
        Initialize rescue router.
        
        Args:
            heightmap: Terrain elevation grid
            safe_havens: List of safe haven POIs (hospitals, police stations)
            flood_depth: Current flood depth grid (optional)
            config: Route configuration
        """
        self.heightmap = heightmap
        self.config = config or RouteConfig()
        self.flood_depth = flood_depth
        
        # Parse safe havens
        self.safe_havens = []
        for haven in safe_havens:
            if haven.get('properties', {}).get('is_safe_haven', False):
                coords = haven.get('geometry', {}).get('coordinates', [])
                if len(coords) >= 2:
                    self.safe_havens.append({
                        'name': haven['properties'].get('name', 'Safe Haven'),
                        'type': haven['properties'].get('type', 'shelter'),
                        'lon': coords[0],
                        'lat': coords[1]
                    })
        
        # Initialize router
        self.weight_calc = TerrainWeightCalculator(heightmap, flood_depth)
        self.router = AStarRouter(self.weight_calc, config)
    
    def _latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to grid position."""
        rows = self.heightmap.shape[0]
        i = int((lat - self.config.center_lat) / self.config.grid_resolution + rows/2)
        j = int((lon - self.config.center_lon) / self.config.grid_resolution + rows/2)
        return (i, j)
    
    def _grid_to_latlon(self, i: int, j: int) -> Tuple[float, float]:
        """Convert grid position to lat/lon."""
        rows = self.heightmap.shape[0]
        lat = self.config.center_lat + (i - rows/2) * self.config.grid_resolution
        lon = self.config.center_lon + (j - rows/2) * self.config.grid_resolution
        return (lat, lon)
    
    def find_nearest_haven(self, user_lat: float, user_lon: float) -> Optional[Dict]:
        """Find the nearest safe haven to user location."""
        if not self.safe_havens:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for haven in self.safe_havens:
            dist = np.sqrt(
                (haven['lat'] - user_lat)**2 + 
                (haven['lon'] - user_lon)**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest = haven
        
        return nearest
    
    def find_rescue_path(self, user_lat: float, user_lon: float,
                         target_haven: Optional[Dict] = None) -> Dict:
        """
        Find safest rescue path from user location to nearest safe haven.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            target_haven: Optional specific haven to route to
            
        Returns:
            GeoJSON FeatureCollection with rescue path and metadata
        """
        # Find target
        if target_haven is None:
            target_haven = self.find_nearest_haven(user_lat, user_lon)
        
        if target_haven is None:
            return {
                "type": "FeatureCollection",
                "features": [],
                "error": "No safe havens available",
                "status": "failed"
            }
        
        # Convert to grid coordinates
        start = self._latlon_to_grid(user_lat, user_lon)
        goal = self._latlon_to_grid(target_haven['lat'], target_haven['lon'])
        
        # Clamp to grid bounds
        rows, cols = self.heightmap.shape
        start = (
            max(0, min(rows-1, start[0])),
            max(0, min(cols-1, start[1]))
        )
        goal = (
            max(0, min(rows-1, goal[0])),
            max(0, min(cols-1, goal[1]))
        )
        
        # Find path
        path = self.router.find_path(start, goal)
        
        if path is None:
            return {
                "type": "FeatureCollection",
                "features": [],
                "error": "No safe path found - area may be flooded",
                "status": "blocked"
            }
        
        # Convert to GeoJSON
        path_feature = self.router.path_to_geojson(path)
        path_feature['properties']['destination'] = target_haven['name']
        path_feature['properties']['destination_type'] = target_haven['type']
        
        # Add start and end markers
        start_marker = {
            "type": "Feature",
            "properties": {
                "type": "user_location",
                "label": "You are here"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [user_lon, user_lat]
            }
        }
        
        end_marker = {
            "type": "Feature",
            "properties": {
                "type": "safe_haven",
                "label": target_haven['name']
            },
            "geometry": {
                "type": "Point",
                "coordinates": [target_haven['lon'], target_haven['lat']]
            }
        }
        
        return {
            "type": "FeatureCollection",
            "features": [path_feature, start_marker, end_marker],
            "status": "success",
            "summary": {
                "distance_km": path_feature['properties']['distance_km'],
                "destination": target_haven['name'],
                "estimated_time_min": round(path_feature['properties']['distance_m'] / 80, 1)  # ~5 km/h walking
            }
        }
    
    def update_flood_state(self, flood_depth: np.ndarray):
        """Update flood depth and recalculate weights."""
        self.flood_depth = flood_depth
        self.weight_calc = TerrainWeightCalculator(self.heightmap, flood_depth)
        self.router = AStarRouter(self.weight_calc, self.config)


def find_rescue_path(user_lat: float, user_lon: float,
                    heightmap: np.ndarray, safe_havens: List[Dict],
                    flood_depth: Optional[np.ndarray] = None,
                    config: RouteConfig = None) -> Dict:
    """
    Convenience function to find rescue path.
    
    Args:
        user_lat: User's latitude
        user_lon: User's longitude
        heightmap: Terrain elevation grid
        safe_havens: List of safe haven POI features
        flood_depth: Optional flood depth grid
        config: Route configuration
        
    Returns:
        GeoJSON FeatureCollection with rescue path
    """
    router = RescueRouter(heightmap, safe_havens, flood_depth, config)
    return router.find_rescue_path(user_lat, user_lon)


if __name__ == "__main__":
    # Test with synthetic data
    from generators import ValleyTerrainGenerator, TerrainConfig, InfrastructureGenerator
    
    # Generate terrain
    config = TerrainConfig()
    terrain = ValleyTerrainGenerator(config)
    heightmap = terrain.generate_heightmap()
    
    # Generate safe havens
    infra_gen = InfrastructureGenerator(terrain)
    pois = infra_gen.generate_pois()
    
    # Find rescue path
    result = find_rescue_path(
        user_lat=11.555,
        user_lon=76.135,
        heightmap=heightmap,
        safe_havens=pois['features']
    )
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        summary = result['summary']
        print(f"Distance: {summary['distance_km']} km")
        print(f"Destination: {summary['destination']}")
        print(f"Estimated time: {summary['estimated_time_min']} minutes")
