"""
Normalize Population Feature

This script adds normalized population feature to the enriched grid features.
Should be run after population_integrator.py.

Module: src/data/normalize_population.py
Feature: 002-population-integration
Created: 2025-12-10
"""

import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_population_feature(
    features_path: str = "data/processed/features/grid_features.parquet",
    scaler_path: str = "models/transit_coverage/scaler.pkl",
    output_path: str = "data/processed/features/grid_features.parquet"
):
    """
    Add normalized population feature to enriched grid features.
    
    Args:
        features_path: Path to features parquet with population column
        scaler_path: Path to save/load population scaler
        output_path: Path to save features with normalized population
    """
    logger.info("=" * 60)
    logger.info("Normalizing Population Feature")
    logger.info("=" * 60)
    
    # Load features
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns")
    
    # Check if population column exists
    if 'population' not in features_df.columns:
        logger.error("'population' column not found in features")
        logger.info(f"Available columns: {', '.join(features_df.columns)}")
        raise ValueError("Population column missing - run population_integrator.py first")
    
    # Fit scaler on population column
    logger.info("Fitting StandardScaler on population column...")
    pop_scaler = StandardScaler()
    population_norm = pop_scaler.fit_transform(features_df[['population']])
    
    # Add normalized column
    features_df['population_norm'] = population_norm
    
    # Log statistics
    logger.info(f"Population statistics:")
    logger.info(f"  Raw - Mean: {features_df['population'].mean():.2f}, Std: {features_df['population'].std():.2f}")
    logger.info(f"  Raw - Min: {features_df['population'].min()}, Max: {features_df['population'].max()}")
    logger.info(f"  Normalized - Mean: {features_df['population_norm'].mean():.6f}, Std: {features_df['population_norm'].std():.6f}")
    logger.info(f"  Normalized - Min: {features_df['population_norm'].min():.6f}, Max: {features_df['population_norm'].max():.6f}")
    
    # Save population scaler
    scaler_dir = Path(scaler_path).parent
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    pop_scaler_path = scaler_dir / "population_scaler.pkl"
    with open(pop_scaler_path, 'wb') as f:
        pickle.dump(pop_scaler, f)
    logger.info(f"Saved population scaler to {pop_scaler_path}")
    
    # Save updated features
    features_df.to_parquet(output_path, compression='snappy')
    file_size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"Saved normalized features to {output_path} ({file_size_kb:.2f} KB)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("POPULATION NORMALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total cells: {len(features_df)}")
    print(f"Raw population: {features_df['population'].min():.0f} - {features_df['population'].max():.0f}")
    print(f"Mean population: {features_df['population'].mean():.1f}")
    print(f"Normalized mean: {features_df['population_norm'].mean():.6f} (target: ~0)")
    print(f"Normalized std: {features_df['population_norm'].std():.6f} (target: ~1)")
    print(f"Columns: {', '.join(features_df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    """
    Normalize population feature in enriched grid features.
    
    Usage:
        python src/data/normalize_population.py
    """
    import sys
    
    try:
        normalize_population_feature()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
