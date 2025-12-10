"""
Feature Extractor for Transit Coverage Classification

This module extracts transit coverage metrics for each grid cell from GTFS data.
Features include stop count, route count, and daily trip frequency.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import yaml
import joblib
from pathlib import Path
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts transit coverage features for grid cells from GTFS data.
    
    Attributes:
        config: Configuration dictionary with feature parameters
        gtfs_data_dir: Directory containing GTFS parquet files
        grid_path: Path to grid parquet file
    """
    
    def __init__(self, 
                 config_path: str = "config/model_config.yaml",
                 gtfs_data_dir: str = "data/processed/gtfs",
                 grid_path: str = "data/processed/grids/cells.parquet"):
        """
        Initialize FeatureExtractor with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
            gtfs_data_dir: Directory with GTFS parquet files
            grid_path: Path to grid parquet file
        """
        self.config = self._load_config(config_path)
        self.gtfs_data_dir = Path(gtfs_data_dir)
        self.grid_path = grid_path
        self.scaler = StandardScaler()
        
        logger.info(f"FeatureExtractor initialized")
        logger.info(f"GTFS data directory: {self.gtfs_data_dir}")
        logger.info(f"Grid path: {self.grid_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_grid(self) -> gpd.GeoDataFrame:
        """Load grid from parquet file."""
        logger.info("Loading grid...")
        grid = gpd.read_parquet(self.grid_path)
        logger.info(f"Loaded {len(grid)} grid cells")
        return grid
    
    def load_gtfs_stops(self) -> gpd.GeoDataFrame:
        """
        Load GTFS stops and convert to GeoDataFrame.
        
        Returns:
            GeoDataFrame with stop locations and geometries
        """
        logger.info("Loading GTFS stops...")
        stops = pd.read_parquet(self.gtfs_data_dir / "stops.parquet")
        
        # Convert to GeoDataFrame
        stops_gdf = gpd.GeoDataFrame(
            stops,
            geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']),
            crs="EPSG:4326"
        )
        
        logger.info(f"Loaded {len(stops_gdf)} stops")
        return stops_gdf
    
    def calculate_stop_count(self, grid: gpd.GeoDataFrame, stops: gpd.GeoDataFrame) -> pd.Series:
        """
        Calculate number of stops within each grid cell using spatial join.
        
        Args:
            grid: Grid GeoDataFrame
            stops: Stops GeoDataFrame
            
        Returns:
            Series with stop counts per cell_id
        """
        logger.info("Calculating stop counts per cell...")
        
        # Spatial join: assign stops to cells
        stops_in_cells = gpd.sjoin(
            stops,
            grid[['cell_id', 'geometry']],
            how='inner',
            predicate='within'
        )
        
        # Count stops per cell
        stop_counts = stops_in_cells.groupby('cell_id').size()
        
        # Fill missing cells with 0
        stop_counts = stop_counts.reindex(grid['cell_id'], fill_value=0)
        
        logger.info(f"Stop count - Mean: {stop_counts.mean():.2f}, "
                   f"Median: {stop_counts.median():.0f}, "
                   f"Max: {stop_counts.max()}")
        
        return stop_counts
    
    def calculate_route_count(self, 
                              grid: gpd.GeoDataFrame, 
                              stops: gpd.GeoDataFrame) -> pd.Series:
        """
        Calculate number of unique routes serving each grid cell.
        
        Args:
            grid: Grid GeoDataFrame
            stops: Stops GeoDataFrame
            
        Returns:
            Series with route counts per cell_id
        """
        logger.info("Calculating route counts per cell...")
        
        # Load stop_times and trips to get route information
        stop_times = pd.read_parquet(self.gtfs_data_dir / "stop_times.parquet")
        trips = pd.read_parquet(self.gtfs_data_dir / "trips.parquet")
        
        # Join stop_times with trips to get route_id for each stop
        stop_routes = stop_times[['stop_id', 'trip_id']].merge(
            trips[['trip_id', 'route_id']],
            on='trip_id'
        )
        
        # Get unique routes per stop
        routes_per_stop = stop_routes.groupby('stop_id')['route_id'].nunique()
        
        # Merge with stops to get geographic information
        stops_with_routes = stops.copy()
        stops_with_routes['route_count'] = stops_with_routes['stop_id'].map(routes_per_stop).fillna(0)
        
        # Spatial join: assign stops to cells
        stops_in_cells = gpd.sjoin(
            stops_with_routes[['stop_id', 'geometry', 'route_count']],
            grid[['cell_id', 'geometry']],
            how='inner',
            predicate='within'
        )
        
        # Get unique routes per cell (sum route counts across stops in cell)
        # Note: This counts routes multiple times if they serve multiple stops in a cell
        # For true unique count, we need to track route IDs per cell
        stop_routes_in_cells = stops_in_cells[['cell_id', 'stop_id']].merge(
            stop_routes[['stop_id', 'route_id']].drop_duplicates(),
            on='stop_id'
        )
        
        route_counts = stop_routes_in_cells.groupby('cell_id')['route_id'].nunique()
        
        # Fill missing cells with 0
        route_counts = route_counts.reindex(grid['cell_id'], fill_value=0)
        
        logger.info(f"Route count - Mean: {route_counts.mean():.2f}, "
                   f"Median: {route_counts.median():.0f}, "
                   f"Max: {route_counts.max()}")
        
        return route_counts
    
    def calculate_daily_trips(self,
                             grid: gpd.GeoDataFrame,
                             stops: gpd.GeoDataFrame) -> pd.Series:
        """
        Calculate daily trip frequency for each grid cell.
        
        Methodology from research.md:
        1. Filter weekday services from calendar
        2. Count unique trips per stop
        3. Aggregate to grid cells
        
        Args:
            grid: Grid GeoDataFrame
            stops: Stops GeoDataFrame
            
        Returns:
            Series with daily trip counts per cell_id
        """
        logger.info("Calculating daily trip frequencies...")
        
        # Load GTFS data
        calendar = pd.read_parquet(self.gtfs_data_dir / "calendar.parquet")
        trips = pd.read_parquet(self.gtfs_data_dir / "trips.parquet")
        stop_times = pd.read_parquet(self.gtfs_data_dir / "stop_times.parquet")
        
        # Filter weekday services (Monday-Friday = 1)
        weekday_services = calendar[
            (calendar['monday'] == 1) &
            (calendar['tuesday'] == 1) &
            (calendar['wednesday'] == 1) &
            (calendar['thursday'] == 1) &
            (calendar['friday'] == 1)
        ]['service_id'].unique()
        
        logger.info(f"Found {len(weekday_services)} weekday service IDs")
        
        # Get trips for weekday services
        weekday_trips = trips[trips['service_id'].isin(weekday_services)]['trip_id'].unique()
        
        logger.info(f"Found {len(weekday_trips)} weekday trips")
        
        # Count trips per stop
        stop_trip_counts = stop_times[
            stop_times['trip_id'].isin(weekday_trips)
        ].groupby('stop_id')['trip_id'].nunique()
        
        # Merge with stops
        stops_with_trips = stops.copy()
        stops_with_trips['daily_trips'] = stops_with_trips['stop_id'].map(stop_trip_counts).fillna(0)
        
        # Spatial join: assign stops to cells
        stops_in_cells = gpd.sjoin(
            stops_with_trips[['stop_id', 'geometry', 'daily_trips']],
            grid[['cell_id', 'geometry']],
            how='inner',
            predicate='within'
        )
        
        # Sum trip counts per cell
        daily_trips = stops_in_cells.groupby('cell_id')['daily_trips'].sum()
        
        # Fill missing cells with 0
        daily_trips = daily_trips.reindex(grid['cell_id'], fill_value=0)
        
        logger.info(f"Daily trips - Mean: {daily_trips.mean():.2f}, "
                   f"Median: {daily_trips.median():.0f}, "
                   f"Max: {daily_trips.max()}")
        
        return daily_trips
    
    def calculate_optional_metrics(self,
                                   grid: gpd.GeoDataFrame,
                                   stop_count: pd.Series,
                                   route_count: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate optional metrics: stop_density and route_diversity.
        
        Args:
            grid: Grid GeoDataFrame
            stop_count: Stop counts per cell
            route_count: Route counts per cell
            
        Returns:
            Tuple of (stop_density, route_diversity) Series
        """
        logger.info("Calculating optional metrics...")
        
        # Stop density: stops per km²
        stop_density = stop_count / grid.set_index('cell_id')['area_km2']
        
        # Route diversity: For now, use route count as diversity measure
        # Could be enhanced with Shannon entropy or other diversity metrics
        route_diversity = route_count.copy()
        
        return stop_density, route_diversity
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extract all features for grid cells.
        
        Returns:
            DataFrame with raw features for all cells
        """
        logger.info("="*60)
        logger.info("Starting feature extraction...")
        logger.info("="*60)
        
        # Load data
        grid = self.load_grid()
        stops = self.load_gtfs_stops()
        
        # Calculate features
        stop_count = self.calculate_stop_count(grid, stops)
        route_count = self.calculate_route_count(grid, stops)
        daily_trips = self.calculate_daily_trips(grid, stops)
        
        # Optional metrics
        if self.config['features'].get('include_density', True):
            stop_density, route_diversity = self.calculate_optional_metrics(
                grid, stop_count, route_count
            )
        
        # Create features DataFrame
        features = pd.DataFrame({
            'cell_id': grid['cell_id'],
            'stop_count': stop_count.values,
            'route_count': route_count.values,
            'daily_trips': daily_trips.values,
        })
        
        if self.config['features'].get('include_density', True):
            features['stop_density'] = stop_density.values
            features['route_diversity'] = route_diversity.values
        
        logger.info(f"Extracted features for {len(features)} cells")
        logger.info(f"Features: {list(features.columns)}")
        
        return features
    
    def normalize_features(self,
                          features: pd.DataFrame,
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler (z-score normalization).
        
        Args:
            features: DataFrame with raw features
            fit: If True, fit scaler on data; if False, use existing scaler
            
        Returns:
            DataFrame with normalized features added
        """
        logger.info("Normalizing features...")
        
        # Features to normalize
        feature_cols = ['stop_count', 'route_count', 'daily_trips']
        
        if fit:
            # Fit scaler on training data
            self.scaler.fit(features[feature_cols])
            logger.info("Fitted StandardScaler on features")
        
        # Transform features
        normalized = self.scaler.transform(features[feature_cols])
        
        # Add normalized features to DataFrame
        for i, col in enumerate(feature_cols):
            features[f'{col}_norm'] = normalized[:, i]
        
        logger.info(f"Normalized features: {[f'{col}_norm' for col in feature_cols]}")
        
        return features
    
    def save_features(self,
                     features: pd.DataFrame,
                     output_path: str = "data/processed/features/grid_features.parquet"):
        """
        Save features to parquet file.
        
        Args:
            features: Features DataFrame
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        features.to_parquet(output_path, index=False)
        
        logger.info(f"Features saved to {output_path}")
        logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    def save_scaler(self, output_path: str = "models/transit_coverage/scaler.pkl"):
        """
        Save fitted scaler for inference.
        
        Args:
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, output_path)
        
        logger.info(f"Scaler saved to {output_path}")
    
    def load_features(self, input_path: str = "data/processed/features/grid_features.parquet") -> pd.DataFrame:
        """
        Load features from parquet file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Features DataFrame
        """
        features = pd.read_parquet(input_path)
        logger.info(f"Loaded features for {len(features)} cells from {input_path}")
        return features


def main():
    """Main function to extract and save features."""
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract raw features
    features = extractor.extract_features()
    
    # Save raw features
    extractor.save_features(features, "data/processed/features/grid_features_raw.parquet")
    
    # Normalize features (fit scaler on all data for now)
    # Note: In actual training, should fit only on training set
    features = extractor.normalize_features(features, fit=True)
    
    # Save normalized features
    extractor.save_features(features, "data/processed/features/grid_features.parquet")
    
    # Save scaler
    extractor.save_scaler()
    
    # Display summary statistics
    print("\n" + "="*60)
    print("Feature Extraction Summary")
    print("="*60)
    print(f"Total cells: {len(features)}")
    print(f"\nRaw Features Statistics:")
    print(features[['stop_count', 'route_count', 'daily_trips']].describe())
    print(f"\nNormalized Features Statistics:")
    print(features[['stop_count_norm', 'route_count_norm', 'daily_trips_norm']].describe())
    print(f"\nCells with zero stops: {(features['stop_count'] == 0).sum()} "
          f"({(features['stop_count'] == 0).sum() / len(features) * 100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
