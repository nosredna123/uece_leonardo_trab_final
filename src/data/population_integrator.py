"""
Population Integrator Module

Loads IBGE population data and merges with transit grid features.

Module: src/data/population_integrator.py
Feature: 002-population-integration
Created: 2025-12-10
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ibge_data(
    zip_path: str,
    bounds: Optional[Dict[str, float]] = None,
    target_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Load IBGE grade estatística population data from ZIP file.
    
    Args:
        zip_path: Path to IBGE ZIP file (e.g., "data/raw/ibge_populacao_bh_grade_id36.zip")
        bounds: Optional geographic bounds to filter data, dict with keys:
                'lat_min', 'lat_max', 'lon_min', 'lon_max'
        target_crs: Target coordinate reference system (default: WGS84)
    
    Returns:
        GeoDataFrame with columns: ['ID36', 'POP', 'geometry']
        - ID36: str, IBGE cell identifier
        - POP: int, population count (>= 0)
        - geometry: Polygon in target_crs
    
    Raises:
        FileNotFoundError: If zip_path doesn't exist
        ValueError: If file format invalid or required columns missing
    
    Example:
        >>> gdf = load_ibge_data("data/raw/ibge_populacao_bh_grade_id36.zip")
        >>> print(f"Loaded {len(gdf)} cells, total pop: {gdf['POP'].sum()}")
    """
    logger.info(f"Loading IBGE data from {zip_path}")
    
    # Validate file exists
    zip_file = Path(zip_path)
    if not zip_file.exists():
        raise FileNotFoundError(
            f"IBGE ZIP file not found: {zip_path}\n"
            f"Remediation:\n"
            f"1. Download from: https://www.ibge.gov.br/estatisticas/downloads-estatisticas.html\n"
            f"2. Look for 'Grade Estatística' > 'Censo 2022' > '200m resolution'\n"
            f"3. Place file at: {zip_path}"
        )
    
    try:
        # Read shapefile from ZIP using PyOGRIO engine (more robust for ZIP archives)
        logger.info(f"Reading shapefile from ZIP archive...")
        gdf = gpd.read_file(f"zip://{zip_path}")
        logger.info(f"Successfully read {len(gdf)} records from ZIP")
        
    except Exception as e:
        raise ValueError(
            f"Failed to read IBGE data from ZIP: {str(e)}\n"
            f"Remediation:\n"
            f"1. Verify ZIP file is not corrupted (try extracting manually)\n"
            f"2. Ensure ZIP contains shapefile (.shp, .dbf, .shx, .prj)\n"
            f"3. Check file encoding (should be UTF-8 or CP1252)"
        )
    
    # Validate required columns - try common IBGE column names
    pop_col = None
    for col_name in ['POP', 'TOTAL', 'populacao', 'pop_total']:
        if col_name in gdf.columns:
            pop_col = col_name
            break
    
    if pop_col is None:
        available_cols = ', '.join(gdf.columns)
        raise ValueError(
            f"Missing required population column (tried: POP, TOTAL, populacao, pop_total) in IBGE data.\n"
            f"Available columns: {available_cols}\n"
            f"Remediation: Verify IBGE data format and column naming"
        )
    
    # Rename to standard 'POP' column
    if pop_col != 'POP':
        logger.info(f"Renaming population column '{pop_col}' -> 'POP'")
        gdf = gdf.rename(columns={pop_col: 'POP'})
    
    # Optionally validate ID36/ID_UNICO column (used for direct merge)
    id_col = None
    for col_name in ['ID36', 'ID_UNICO', 'id_unico', 'id']:
        if col_name in gdf.columns:
            id_col = col_name
            break
    
    has_id36 = id_col is not None
    if has_id36 and id_col != 'ID36':
        logger.info(f"Renaming ID column '{id_col}' -> 'ID36'")
        gdf = gdf.rename(columns={id_col: 'ID36'})
    elif not has_id36:
        logger.warning("No ID column found (tried: ID36, ID_UNICO, id_unico, id) - will use spatial join only for merge")
    
    # Validate population values
    if gdf['POP'].isna().any():
        logger.warning(f"Found {gdf['POP'].isna().sum()} null population values - filling with 0")
        gdf['POP'] = gdf['POP'].fillna(0)
    
    if (gdf['POP'] < 0).any():
        invalid_count = (gdf['POP'] < 0).sum()
        raise ValueError(
            f"Found {invalid_count} negative population values.\n"
            f"Population must be non-negative integers."
        )
    
    # Ensure population is integer type
    gdf['POP'] = gdf['POP'].astype(int)
    
    # Validate geometries
    invalid_geoms = ~gdf.geometry.is_valid
    if invalid_geoms.any():
        invalid_count = invalid_geoms.sum()
        logger.warning(f"Found {invalid_count} invalid geometries - attempting to fix")
        gdf.geometry = gdf.geometry.buffer(0)  # Fix common topology issues
        
        # Re-check after fix
        still_invalid = ~gdf.geometry.is_valid
        if still_invalid.any():
            raise ValueError(
                f"Failed to fix {still_invalid.sum()} invalid geometries.\n"
                f"Problematic record IDs: {gdf[still_invalid].index.tolist()[:10]}"
            )
    
    # Apply geographic bounds filter if provided
    if bounds is not None:
        logger.info(f"Filtering to geographic bounds: {bounds}")
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        
        # Filter by centroid coordinates
        centroids = gdf.geometry.centroid
        mask = (
            (centroids.y >= lat_min) &
            (centroids.y <= lat_max) &
            (centroids.x >= lon_min) &
            (centroids.x <= lon_max)
        )
        gdf = gdf[mask].copy()
        logger.info(f"Retained {len(gdf)} cells within bounds")
    
    # Reproject to target CRS if needed
    if gdf.crs is None:
        logger.warning(f"No CRS found in IBGE data - assuming {target_crs}")
        gdf = gdf.set_crs(target_crs)
    elif str(gdf.crs) != target_crs:
        logger.info(f"Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    
    # Select and order columns
    columns = ['geometry', 'POP']
    if has_id36:
        columns.insert(0, 'ID36')
    gdf = gdf[columns].copy()
    
    # Log summary statistics
    total_pop = gdf['POP'].sum()
    zero_pop_pct = (gdf['POP'] == 0).mean() * 100
    logger.info(f"Loaded {len(gdf)} IBGE cells")
    logger.info(f"Total population: {total_pop:,}")
    logger.info(f"Cells with zero population: {zero_pop_pct:.1f}%")
    logger.info(f"Population range: {gdf['POP'].min()} - {gdf['POP'].max()}")
    logger.info(f"CRS: {gdf.crs}")
    
    return gdf


def merge_population(
    grid_gdf: gpd.GeoDataFrame,
    ibge_gdf: gpd.GeoDataFrame,
    method: str = "auto"
) -> gpd.GeoDataFrame:
    """
    Merge IBGE population data with transit grid features.
    
    Args:
        grid_gdf: GeoDataFrame with transit features, must have 'cell_id' and 'geometry'
        ibge_gdf: GeoDataFrame from load_ibge_data(), must have 'POP' and 'geometry'
        method: Merge method - "id" (direct ID36 merge), "spatial" (spatial join),
                or "auto" (try ID first, fallback to spatial)
    
    Returns:
        GeoDataFrame with all grid_gdf columns plus:
        - population: int, population count (0 if unmatched)
        - merge_method: str, "id" or "spatial" (indicates how matched)
    
    Raises:
        ValueError: If required columns missing or CRS mismatch unresolvable
    
    Example:
        >>> grid = gpd.read_parquet("data/processed/features/grid_features.parquet")
        >>> ibge = load_ibge_data("data/raw/ibge_populacao_bh_grade_id36.zip")
        >>> enriched = merge_population(grid, ibge, method="auto")
        >>> coverage = (enriched['population'] > 0).mean()
        >>> print(f"Population coverage: {coverage:.1%}")
    """
    logger.info(f"Merging population data using method: {method}")
    logger.info(f"Grid cells: {len(grid_gdf)}, IBGE cells: {len(ibge_gdf)}")
    
    # Validate required columns
    if 'cell_id' not in grid_gdf.columns:
        raise ValueError("grid_gdf missing required column 'cell_id'")
    if 'POP' not in ibge_gdf.columns:
        raise ValueError("ibge_gdf missing required column 'POP'")
    
    # Ensure CRS match
    if grid_gdf.crs != ibge_gdf.crs:
        logger.info(f"CRS mismatch - reprojecting IBGE data from {ibge_gdf.crs} to {grid_gdf.crs}")
        ibge_gdf = ibge_gdf.to_crs(grid_gdf.crs)
    
    # Initialize result
    result_gdf = grid_gdf.copy()
    result_gdf['population'] = 0  # Default to 0 for unmatched cells
    result_gdf['merge_method'] = None
    
    # Attempt ID-based merge first if using "auto" or "id" method
    id_merge_successful = False
    if method in ["auto", "id"]:
        if 'ID36' in ibge_gdf.columns and 'ID36' in grid_gdf.columns:
            logger.info("Attempting ID-based merge on 'ID36' column...")
            
            # Perform left join on ID36
            id_merged = result_gdf.merge(
                ibge_gdf[['ID36', 'POP']],
                on='ID36',
                how='left',
                suffixes=('', '_ibge')
            )
            
            # Count successful matches
            matched_mask = id_merged['POP'].notna()
            match_count = matched_mask.sum()
            match_pct = (match_count / len(grid_gdf)) * 100
            
            logger.info(f"ID merge matched {match_count}/{len(grid_gdf)} cells ({match_pct:.1f}%)")
            
            if match_count > 0:
                # Update population for matched cells
                result_gdf.loc[matched_mask, 'population'] = id_merged.loc[matched_mask, 'POP'].astype(int)
                result_gdf.loc[matched_mask, 'merge_method'] = 'id'
                id_merge_successful = True
                
                # If method is "id" only, we're done
                if method == "id":
                    logger.info(f"ID merge complete - {match_count} cells matched")
                    return result_gdf
                
                # For "auto", continue with spatial join for unmatched cells
                if method == "auto" and match_pct < 100:
                    unmatched_count = len(grid_gdf) - match_count
                    logger.info(f"Proceeding with spatial join for {unmatched_count} unmatched cells...")
        else:
            logger.warning("ID36 column not found in both datasets - skipping ID merge")
    
    # Perform spatial join (for "spatial" method or "auto" fallback)
    if method == "spatial" or (method == "auto" and not id_merge_successful):
        logger.info("Performing spatial join (centroid-in-polygon)...")
        
        # Get cells that still need population (if coming from auto mode)
        if method == "auto" and id_merge_successful:
            cells_to_join = result_gdf[result_gdf['population'] == 0].copy()
        else:
            cells_to_join = result_gdf.copy()
        
        # Calculate centroids in projected CRS to avoid geographic coordinate warning
        # Use UTM Zone 23S (EPSG:31983) for Belo Horizonte
        is_geographic = cells_to_join.crs.is_geographic if cells_to_join.crs else True
        
        if is_geographic:
            # Temporarily reproject to UTM for accurate centroid calculation
            utm_crs = "EPSG:31983"  # UTM Zone 23S for Belo Horizonte
            cells_utm = cells_to_join.to_crs(utm_crs)
            centroids_utm = cells_utm.geometry.centroid
            # Reproject centroids back to original CRS
            cells_to_join['centroid_geom'] = centroids_utm.to_crs(cells_to_join.crs)
            logger.info(f"Calculated centroids in projected CRS ({utm_crs}) for accuracy")
        else:
            # Already in projected CRS, can calculate directly
            cells_to_join['centroid_geom'] = cells_to_join.geometry.centroid
        
        # Create temporary GeoDataFrame with centroids as geometry
        cells_centroids = gpd.GeoDataFrame(
            cells_to_join.drop(columns=['geometry']),
            geometry='centroid_geom',
            crs=cells_to_join.crs
        )
        
        # Spatial join: find which IBGE cell contains each grid centroid
        joined = gpd.sjoin(
            cells_centroids,
            ibge_gdf[['geometry', 'POP']],
            how='left',
            predicate='within'
        )
        
        # Handle multiple matches (keep first match, sum if multiple)
        if 'index_right' in joined.columns:
            # Check for duplicate matches
            duplicates = joined[joined.duplicated(subset='cell_id', keep=False)]
            if len(duplicates) > 0:
                logger.warning(f"Found {len(duplicates)} cells with multiple IBGE matches - summing population")
                # Group by cell_id and sum population
                pop_sums = joined.groupby('cell_id')['POP'].sum().reset_index()
            else:
                # No duplicates - just use first match
                pop_sums = joined[['cell_id', 'POP']].drop_duplicates(subset='cell_id')
        else:
            pop_sums = joined[['cell_id', 'POP']].drop_duplicates(subset='cell_id')
        
        # Update population for matched cells
        matched_mask = pop_sums['POP'].notna()
        match_count = matched_mask.sum()
        match_pct = (match_count / len(cells_to_join)) * 100
        
        logger.info(f"Spatial join matched {match_count}/{len(cells_to_join)} cells ({match_pct:.1f}%)")
        
        if match_count > 0:
            # Merge back to result
            for idx, row in pop_sums[matched_mask].iterrows():
                cell_mask = result_gdf['cell_id'] == row['cell_id']
                if result_gdf.loc[cell_mask, 'population'].iloc[0] == 0:  # Only update if not already set by ID merge
                    result_gdf.loc[cell_mask, 'population'] = int(row['POP'])
                    result_gdf.loc[cell_mask, 'merge_method'] = 'spatial'
    
    # Final statistics
    total_pop = result_gdf['population'].sum()
    coverage_pct = (result_gdf['population'] > 0).mean() * 100
    zero_pop_pct = (result_gdf['population'] == 0).mean() * 100
    
    logger.info(f"Merge complete:")
    logger.info(f"  Total population: {total_pop:,}")
    logger.info(f"  Cells with population: {coverage_pct:.1f}%")
    logger.info(f"  Cells with zero population: {zero_pop_pct:.1f}%")
    logger.info(f"  Population range: {result_gdf['population'].min()} - {result_gdf['population'].max()}")
    
    if 'merge_method' in result_gdf.columns:
        method_counts = result_gdf['merge_method'].value_counts()
        logger.info(f"  Merge methods: {method_counts.to_dict()}")
    
    return result_gdf


def validate_population_data(
    enriched_gdf: gpd.GeoDataFrame,
    expected_total_pop: Optional[int] = None,
    min_coverage_pct: float = 95.0,
    max_zero_pop_pct: float = 10.0
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate merged population data against quality thresholds.
    
    Args:
        enriched_gdf: GeoDataFrame from merge_population() with 'population' column
        expected_total_pop: Expected total population (optional, for sanity check)
        min_coverage_pct: Minimum % of cells that must have population data
        max_zero_pop_pct: Maximum % of cells allowed to have zero population
    
    Returns:
        Tuple of (is_valid: bool, stats: dict)
        - is_valid: True if all validation checks pass
        - stats: Dictionary with validation results and statistics
    
    Example:
        >>> is_valid, stats = validate_population_data(enriched_gdf, expected_total_pop=2_400_000)
        >>> if not is_valid:
        >>>     print(f"Validation failed: {stats['errors']}")
    """
    logger.info("Validating merged population data...")
    
    # Initialize validation results
    stats = {
        'cell_count': len(enriched_gdf),
        'total_population': 0,
        'coverage_pct': 0.0,
        'zero_pop_pct': 0.0,
        'pop_range': (0, 0),
        'errors': [],
        'warnings': []
    }
    
    # Validate required column
    if 'population' not in enriched_gdf.columns:
        stats['errors'].append("Missing required column 'population'")
        return False, stats
    
    # Calculate statistics
    total_pop = enriched_gdf['population'].sum()
    coverage_pct = (enriched_gdf['population'] > 0).mean() * 100
    zero_pop_pct = (enriched_gdf['population'] == 0).mean() * 100
    pop_min = enriched_gdf['population'].min()
    pop_max = enriched_gdf['population'].max()
    
    stats['total_population'] = int(total_pop)
    stats['coverage_pct'] = round(coverage_pct, 2)
    stats['zero_pop_pct'] = round(zero_pop_pct, 2)
    stats['pop_range'] = (int(pop_min), int(pop_max))
    
    # Validation checks
    is_valid = True
    
    # Check 1: Minimum coverage
    if coverage_pct < min_coverage_pct:
        is_valid = False
        stats['errors'].append(
            f"Coverage {coverage_pct:.1f}% below minimum {min_coverage_pct}%"
        )
    else:
        logger.info(f"✓ Coverage check passed: {coverage_pct:.1f}% >= {min_coverage_pct}%")
    
    # Check 2: Maximum zero population percentage
    if zero_pop_pct > max_zero_pop_pct:
        stats['warnings'].append(
            f"Zero population cells {zero_pop_pct:.1f}% exceeds threshold {max_zero_pop_pct}%"
        )
        logger.warning(f"⚠ High zero-population rate: {zero_pop_pct:.1f}%")
    else:
        logger.info(f"✓ Zero-population check passed: {zero_pop_pct:.1f}% <= {max_zero_pop_pct}%")
    
    # Check 3: Expected total population (if provided)
    if expected_total_pop is not None:
        tolerance = 0.10  # 10% tolerance
        lower_bound = expected_total_pop * (1 - tolerance)
        upper_bound = expected_total_pop * (1 + tolerance)
        
        if not (lower_bound <= total_pop <= upper_bound):
            stats['warnings'].append(
                f"Total population {total_pop:,} outside expected range "
                f"[{lower_bound:,.0f}, {upper_bound:,.0f}]"
            )
            logger.warning(f"⚠ Total population {total_pop:,} differs from expected {expected_total_pop:,}")
        else:
            logger.info(f"✓ Total population check passed: {total_pop:,} ≈ {expected_total_pop:,}")
    
    # Check 4: Non-negative values
    if pop_min < 0:
        is_valid = False
        stats['errors'].append(f"Found negative population values (min: {pop_min})")
    else:
        logger.info(f"✓ Non-negative check passed: min = {pop_min}")
    
    # Check 5: Reasonable maximum per cell (200m grid)
    max_reasonable_pop = 3000  # 200m cells should have lower max than 250m
    if pop_max > max_reasonable_pop:
        stats['warnings'].append(
            f"Maximum population {pop_max} exceeds reasonable threshold {max_reasonable_pop}"
        )
        logger.warning(f"⚠ High maximum population: {pop_max} (expected <{max_reasonable_pop})")
    else:
        logger.info(f"✓ Maximum population check passed: {pop_max} <= {max_reasonable_pop}")
    
    # Summary
    if is_valid:
        logger.info("✓ Validation PASSED")
    else:
        logger.error(f"✗ Validation FAILED: {len(stats['errors'])} errors")
        for error in stats['errors']:
            logger.error(f"  - {error}")
    
    if stats['warnings']:
        logger.warning(f"Validation has {len(stats['warnings'])} warnings:")
        for warning in stats['warnings']:
            logger.warning(f"  - {warning}")
    
    return is_valid, stats


if __name__ == "__main__":
    """
    Test script for population integrator module.
    
    Usage:
        python src/data/population_integrator.py
    
    This will:
    1. Load IBGE data from data/raw/ibge_populacao_bh_grade_id36.zip
    2. Load grid features from data/processed/features/grid_features.parquet
    3. Merge population data
    4. Validate results
    5. Save enriched dataset to data/processed/features/grid_features_with_population.parquet
    """
    import sys
    
    # File paths
    ibge_zip = "data/raw/ibge_populacao_bh_grade_id36.zip"
    grid_cells = "data/processed/grids/cells.parquet"  # Original grid with geometry
    grid_features = "data/processed/features/grid_features.parquet"  # Transit features (no geometry)
    output_path = "data/processed/features/grid_features_with_population.parquet"
    
    try:
        # Step 1: Load IBGE data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading IBGE population data")
        logger.info("=" * 60)
        ibge_gdf = load_ibge_data(ibge_zip)
        
        # Step 2: Load grid with geometry and features
        logger.info("=" * 60)
        logger.info("STEP 2: Loading grid and features")
        logger.info("=" * 60)
        
        # Load original grid with geometry
        grid_gdf = gpd.read_parquet(grid_cells)
        logger.info(f"Loaded {len(grid_gdf)} grid cells with geometry")
        
        # Load transit features
        features_df = pd.read_parquet(grid_features)
        logger.info(f"Loaded transit features: {', '.join(features_df.columns)}")
        
        # Merge geometry with features (on cell_id)
        grid_gdf = grid_gdf.merge(features_df, on='cell_id', how='left')
        logger.info(f"Merged grid has {len(grid_gdf)} cells with {len(grid_gdf.columns)} columns")
        
        # Step 3: Merge population
        logger.info("=" * 60)
        logger.info("STEP 3: Merging population data")
        logger.info("=" * 60)
        enriched_gdf = merge_population(grid_gdf, ibge_gdf, method="auto")
        
        # Step 4: Validate
        logger.info("=" * 60)
        logger.info("STEP 4: Validating merged data")
        logger.info("=" * 60)
        is_valid, stats = validate_population_data(
            enriched_gdf,
            expected_total_pop=None,  # Don't validate total - IBGE data covers larger area
            min_coverage_pct=50.0,  # More realistic for 200m grid with 2km buffer
            max_zero_pop_pct=50.0   # Allow higher zero-pop rate for fine-grained grid
        )
        
        # Step 5: Save results
        if is_valid or len(stats['errors']) == 0:
            logger.info("=" * 60)
            logger.info("STEP 5: Saving enriched dataset")
            logger.info("=" * 60)
            
            # Drop geometry and merge_method columns for downstream compatibility
            output_df = enriched_gdf.drop(columns=['geometry', 'merge_method'], errors='ignore')
            if isinstance(output_df, gpd.GeoDataFrame):
                output_df = pd.DataFrame(output_df)
            
            output_df.to_parquet(output_path, compression='snappy')
            file_size_kb = Path(output_path).stat().st_size / 1024
            logger.info(f"✓ Saved to {output_path} ({file_size_kb:.2f} KB)")
            logger.info(f"Columns: {', '.join(output_df.columns)}")
            
            # Print summary
            print("\n" + "=" * 60)
            print("POPULATION INTEGRATION SUMMARY")
            print("=" * 60)
            print(f"Total cells: {stats['cell_count']}")
            print(f"Total population: {stats['total_population']:,}")
            print(f"Coverage: {stats['coverage_pct']:.1f}%")
            print(f"Zero-population cells: {stats['zero_pop_pct']:.1f}%")
            print(f"Population range: {stats['pop_range'][0]:,} - {stats['pop_range'][1]:,}")
            print("=" * 60)
            
            sys.exit(0)
        else:
            logger.error("Validation failed - not saving results")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
