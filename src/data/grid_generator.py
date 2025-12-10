"""
Grid Generator for Transit Coverage Classification

This module generates a geographic grid covering the Belo Horizonte transit service area.
Grid cells are square/rectangular regions aligned with lat/lon coordinates.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import yaml
from pathlib import Path
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridGenerator:
    """
    Generates geographic grid for transit coverage analysis.
    
    Attributes:
        config: Configuration dictionary with grid parameters
        bounds: Geographic boundaries (lat_min, lat_max, lon_min, lon_max)
        cell_size_meters: Size of each grid cell in meters
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize GridGenerator with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.bounds = self.config['grid']['bounds']
        self.cell_size_meters = self.config['grid']['cell_size_meters']
        
        logger.info(f"GridGenerator initialized with {self.cell_size_meters}m cells")
        logger.info(f"Bounds: lat [{self.bounds['lat_min']:.6f}, {self.bounds['lat_max']:.6f}], "
                   f"lon [{self.bounds['lon_min']:.6f}, {self.bounds['lon_max']:.6f}]")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _meters_to_degrees_lat(self, meters: float) -> float:
        """
        Convert meters to degrees latitude.
        
        1 degree latitude ≈ 111,320 meters (constant)
        
        Args:
            meters: Distance in meters
            
        Returns:
            Distance in degrees latitude
        """
        return meters / 111320.0
    
    def _meters_to_degrees_lon(self, meters: float, latitude: float) -> float:
        """
        Convert meters to degrees longitude at given latitude.
        
        1 degree longitude = 111,320 * cos(latitude) meters
        
        Args:
            meters: Distance in meters
            latitude: Reference latitude in degrees
            
        Returns:
            Distance in degrees longitude
        """
        return meters / (111320.0 * abs(np.cos(np.radians(latitude))))
    
    def generate_grid(self) -> gpd.GeoDataFrame:
        """
        Generate geographic grid covering the specified bounds.
        
        Returns:
            GeoDataFrame with grid cells containing:
                - cell_id: Unique identifier (format: "cell_{row}_{col}")
                - lat_min, lat_max: Latitude bounds
                - lon_min, lon_max: Longitude bounds
                - centroid_lat, centroid_lon: Cell center coordinates
                - area_km2: Cell area in square kilometers
                - geometry: Shapely Polygon geometry
        """
        logger.info("Starting grid generation...")
        
        # Calculate cell size in degrees
        cell_size_lat = self._meters_to_degrees_lat(self.cell_size_meters)
        
        # Use average latitude for longitude calculation
        avg_lat = (self.bounds['lat_min'] + self.bounds['lat_max']) / 2
        cell_size_lon = self._meters_to_degrees_lon(self.cell_size_meters, avg_lat)
        
        logger.info(f"Cell size: {cell_size_lat:.6f}° lat, {cell_size_lon:.6f}° lon (at {avg_lat:.2f}° lat)")
        
        # Generate grid coordinates
        lat_edges = np.arange(
            self.bounds['lat_min'],
            self.bounds['lat_max'] + cell_size_lat,
            cell_size_lat
        )
        lon_edges = np.arange(
            self.bounds['lon_min'],
            self.bounds['lon_max'] + cell_size_lon,
            cell_size_lon
        )
        
        # Create grid cells
        cells = []
        cell_id = 0
        
        for row, lat_min in enumerate(lat_edges[:-1]):
            lat_max = lat_edges[row + 1]
            
            for col, lon_min in enumerate(lon_edges[:-1]):
                lon_max = lon_edges[col + 1]
                
                # Create cell geometry
                geometry = box(lon_min, lat_min, lon_max, lat_max)
                
                # Calculate centroid
                centroid_lat = (lat_min + lat_max) / 2
                centroid_lon = (lon_min + lon_max) / 2
                
                # Calculate area (approximate, assuming small cells)
                # Convert degrees to km
                height_km = (lat_max - lat_min) * 111.32
                width_km = (lon_max - lon_min) * 111.32 * abs(np.cos(np.radians(centroid_lat)))
                area_km2 = height_km * width_km
                
                cells.append({
                    'cell_id': f'cell_{row}_{col}',
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lon_min': lon_min,
                    'lon_max': lon_max,
                    'centroid_lat': centroid_lat,
                    'centroid_lon': centroid_lon,
                    'area_km2': area_km2,
                    'geometry': geometry
                })
        
        # Create GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
        
        logger.info(f"Generated {len(grid_gdf)} grid cells")
        logger.info(f"Grid dimensions: {len(lat_edges)-1} rows × {len(lon_edges)-1} cols")
        
        return grid_gdf
    
    def validate_grid(self, grid_gdf: gpd.GeoDataFrame) -> bool:
        """
        Validate generated grid meets expectations.
        
        Args:
            grid_gdf: Generated grid GeoDataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating grid...")
        
        expected_cells = self.config['grid'].get('expected_cells', 3520)
        expected_area = self.config['grid'].get('expected_area_km2', 0.25)
        
        # Check cell count (allow 10% deviation)
        cell_count_ok = abs(len(grid_gdf) - expected_cells) / expected_cells <= 0.1
        logger.info(f"Cell count: {len(grid_gdf)} (expected ~{expected_cells}) - "
                   f"{'✓ PASS' if cell_count_ok else '✗ FAIL'}")
        
        # Check average cell area (allow 10% deviation)
        avg_area = grid_gdf['area_km2'].mean()
        area_ok = abs(avg_area - expected_area) / expected_area <= 0.1
        logger.info(f"Average cell area: {avg_area:.3f} km² (expected ~{expected_area} km²) - "
                   f"{'✓ PASS' if area_ok else '✗ FAIL'}")
        
        # Check for valid geometries
        geometries_ok = grid_gdf['geometry'].is_valid.all()
        logger.info(f"All geometries valid: {'✓ PASS' if geometries_ok else '✗ FAIL'}")
        
        # Check for no null values in critical columns
        null_check = grid_gdf[['cell_id', 'lat_min', 'lat_max', 'lon_min', 'lon_max']].isnull().any().any()
        null_ok = not null_check
        logger.info(f"No null values in critical columns: {'✓ PASS' if null_ok else '✗ FAIL'}")
        
        validation_passed = cell_count_ok and area_ok and geometries_ok and null_ok
        
        if validation_passed:
            logger.info("✓ Grid validation PASSED")
        else:
            logger.warning("✗ Grid validation FAILED")
        
        return validation_passed
    
    def save_grid(self, grid_gdf: gpd.GeoDataFrame, output_path: str = "data/processed/grids/cells.parquet"):
        """
        Save grid to parquet file.
        
        Args:
            grid_gdf: Grid GeoDataFrame to save
            output_path: Output file path
        """
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        grid_gdf.to_parquet(output_path)
        
        logger.info(f"Grid saved to {output_path}")
        logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    def load_grid(self, input_path: str = "data/processed/grids/cells.parquet") -> gpd.GeoDataFrame:
        """
        Load grid from parquet file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Grid GeoDataFrame
        """
        grid_gdf = gpd.read_parquet(input_path)
        logger.info(f"Loaded grid with {len(grid_gdf)} cells from {input_path}")
        return grid_gdf


def main():
    """Main function to generate and save grid."""
    # Initialize generator
    generator = GridGenerator()
    
    # Generate grid
    grid = generator.generate_grid()
    
    # Validate grid
    is_valid = generator.validate_grid(grid)
    
    if not is_valid:
        logger.warning("Grid validation failed, but continuing with save...")
    
    # Save grid
    generator.save_grid(grid)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("Grid Generation Summary")
    print("="*60)
    print(f"Total cells: {len(grid)}")
    print(f"Cell area: {grid['area_km2'].mean():.3f} km² (avg)")
    print(f"Latitude range: [{grid['lat_min'].min():.6f}, {grid['lat_max'].max():.6f}]")
    print(f"Longitude range: [{grid['lon_min'].min():.6f}, {grid['lon_max'].max():.6f}]")
    print("\nFirst 5 cells:")
    print(grid[['cell_id', 'centroid_lat', 'centroid_lon', 'area_km2']].head())
    print("="*60)


if __name__ == "__main__":
    main()
