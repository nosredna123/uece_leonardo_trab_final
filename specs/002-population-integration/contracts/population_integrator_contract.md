# Module Contract: population_integrator.py

**Module**: `src/data/population_integrator.py`  
**Purpose**: Load IBGE population data and merge with transit grid features  
**Created**: 2025-12-10  
**Feature**: 002-population-integration

---

## Public API

### Function: `load_ibge_data`

**Signature**:
```python
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
```

**Behavior**:
- Reads shapefile directly from ZIP using PyOGRIO engine
- Validates required column `POP` exists (strict match, case-sensitive)
- Optionally validates `ID36` column (used for direct merge if present)
- Filters to geographic bounds if provided
- Reprojects to target_crs if source CRS differs
- Validates population values: non-negative integers
- Logs summary: cell count, total population, CRS transformation

**Error Handling**:
- File not found → Raise `FileNotFoundError` with remediation steps (FR-004)
- Corrupted ZIP → Raise `ValueError` with cause and remediation
- Missing `POP` column → Raise `ValueError` listing available columns
- Invalid geometries → Raise `ValueError` identifying problematic records

**Performance**:
- Expected runtime: 2-5 seconds for typical file size (5-20 MB)
- Memory usage: ~50-100 MB in-memory GeoDataFrame

---

### Function: `merge_population`

**Signature**:
```python
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
        >>> grid = gpd.read_parquet("data/processed/features/grid_features_transit_only.parquet")
        >>> ibge = load_ibge_data("data/raw/ibge_populacao_bh_grade_id36.zip")
        >>> enriched = merge_population(grid, ibge, method="auto")
        >>> coverage = (enriched['population'] > 0).mean()
        >>> print(f"Population coverage: {coverage:.1%}")
    """
```

**Behavior**:
- Validates CRS compatibility, reprojects if needed
- **Method "auto"** (default):
  1. Check if both datasets have `ID36` column
  2. If yes, attempt direct merge on `ID36`
  3. If match rate <95%, log warning and fall back to spatial join
  4. If no `ID36`, use spatial join
- **Method "id"**: Direct merge on `ID36`, raise error if column missing
- **Method "spatial"**: Spatial join using centroid intersection
- Fills unmatched cells with `population = 0`
- Logs match statistics: coverage rate, unmatched cells, merge method used
- Validates result: coverage >95% (SC-003), total population within expected range

**Error Handling**:
- Missing required columns → Raise `ValueError` with column requirements
- CRS mismatch → Auto-reproject with warning, or raise if ambiguous
- Low coverage (<95%) → Log warning but continue (user should investigate)

**Performance**:
- Direct ID merge: ~1 second for 15k cells
- Spatial join: ~15-20 seconds for 15k cells
- Memory overhead: ~30 MB for merge operation

---

### Function: `validate_population_data`

**Signature**:
```python
def validate_population_data(
    enriched_gdf: gpd.GeoDataFrame,
    expected_total_pop: Tuple[int, int] = (2_000_000, 2_800_000),
    min_coverage: float = 0.95,
    max_zero_pop_pct: float = 0.10
) -> Dict[str, Any]:
    """
    Validate merged population data quality.
    
    Args:
        enriched_gdf: GeoDataFrame with 'population' column
        expected_total_pop: Tuple of (min, max) expected total population
        min_coverage: Minimum fraction of cells with population data
        max_zero_pop_pct: Maximum allowed percentage of zero-population cells
    
    Returns:
        Dictionary with validation results:
        {
            "total_population": int,
            "total_cells": int,
            "populated_cells": int,
            "zero_pop_cells": int,
            "zero_pop_pct": float,
            "min_pop": int,
            "max_pop": int,
            "mean_pop_all": float,
            "mean_pop_populated": float,
            "median_pop": float,
            "coverage_rate": float,
            "validation_passed": bool,
            "validation_warnings": List[str]
        }
    
    Raises:
        ValueError: If critical validation fails (coverage too low or total pop out of range)
    
    Example:
        >>> stats = validate_population_data(enriched)
        >>> print(f"Mean population: {stats['mean_pop_populated']:.1f} per cell")
        >>> if not stats['validation_passed']:
        ...     for warning in stats['validation_warnings']:
        ...         print(f"WARNING: {warning}")
    """
```

**Behavior**:
- Calculates comprehensive statistics per FR-012
- Checks against all success criteria: SC-003, SC-004, SC-005, SC-010
- Generates warnings for violations, raises errors for critical failures
- Returns structured results for logging and reporting

**Validation rules**:
- CRITICAL: Total population within expected range → Error if violated (SC-010)
- CRITICAL: Coverage rate ≥ min_coverage → Error if violated (SC-003)
- WARNING: Zero-population % ≤ max_zero_pop_pct → Warning if violated (SC-005)
- WARNING: Mean population within realistic range [60, 400] → Warning if violated (SC-004)

---

## Data Contracts

### Input Contract: IBGE ZIP File

**Expected structure**:
```
ibge_populacao_bh_grade_id36.zip
└── [shapefile_name].shp    # Primary file
    [shapefile_name].dbf    # Attribute table (contains POP column)
    [shapefile_name].shx    # Spatial index
    [shapefile_name].prj    # Projection info
```

**Required attributes** (in .dbf):
- `POP`: Integer, population count, non-negative
- `geometry`: Polygon, 200m nominal resolution
- `ID36`: String (optional but recommended), IBGE cell identifier

**Coordinate system**: SIRGAS 2000 (EPSG:4674) or WGS84 (EPSG:4326)

**Quality expectations**:
- Total population: 2.0M - 2.8M for Belo Horizonte region
- Cell count: 14,000 - 22,000 cells
- Valid geometries: No null or invalid polygons

---

### Output Contract: Enriched Grid Features

**File**: `data/processed/features/grid_features.parquet`

**Schema**:
```python
{
    # Grid identification (preserved from input)
    "cell_id": str,
    "geometry": Polygon,
    "centroid_lat": float,
    "centroid_lon": float,
    "area_km2": float,
    
    # Transit features (preserved from input)
    "stop_count": int,
    "route_count": int,
    "daily_trips": int,
    "stop_density": float,
    "route_diversity": float,
    
    # Population feature (NEW)
    "population": int,         # ≥ 0, added by merge_population()
    "merge_method": str,       # "id" or "spatial", metadata for debugging
}
```

**Quality guarantees**:
- All cells from input grid_gdf preserved (left join)
- `population` column: Non-null, non-negative integers
- Coverage: >95% cells have population > 0 (SC-003)
- Total population sum: Within expected range ±10% (SC-010)
- File format: Parquet with snappy compression
- File size: Baseline + <15% (SC-008)

---

## Integration Points

### Upstream Dependencies

1. **Grid Generator** (`src/data/grid_generator.py`):
   - Must generate 200m grid (config: `cell_size_meters = 200`)
   - Must output Parquet with `cell_id`, `geometry`, `centroid_lat`, `centroid_lon`
   - Must use EPSG:4326 CRS for consistency

2. **Feature Extractor** (`src/data/feature_extractor.py`):
   - Must extract transit features for 200m grid
   - Must preserve all grid cells in output (no filtering)
   - Must output GeoDataFrame or Parquet with geometry column

3. **Configuration** (`config/model_config.yaml`):
   - Must define `features.population.source_file` path
   - Must define geographic bounds for filtering
   - Must define validation thresholds

---

### Downstream Dependencies

1. **Model Training** (`src/models/train.py`):
   - Must include `population` in feature set
   - Must apply StandardScaler to all features including population
   - Must handle zero-population cells appropriately (include in training)

2. **Report Generation** (`generate_report.py`):
   - Must document population integration in technical report
   - Must display population statistics from `validate_population_data()`
   - Must show feature importance including population

3. **Pipeline Script** (`run_pipeline.sh`):
   - Must call population integration after feature extraction
   - Must validate integration success before proceeding to training
   - Must log population statistics

---

## Usage Example

```python
from src.data.population_integrator import (
    load_ibge_data, 
    merge_population, 
    validate_population_data
)
import geopandas as gpd

# Load grid with transit features
grid = gpd.read_parquet("data/processed/features/grid_features_transit_only.parquet")
print(f"Grid cells: {len(grid)}")

# Load IBGE population data
ibge = load_ibge_data(
    zip_path="data/raw/ibge_populacao_bh_grade_id36.zip",
    bounds={
        'lat_min': -20.046411,
        'lat_max': -19.758246,
        'lon_min': -44.081380,
        'lon_max': -43.843522
    }
)
print(f"IBGE cells: {len(ibge)}, Total pop: {ibge['POP'].sum():,}")

# Merge population with grid
enriched = merge_population(grid, ibge, method="auto")
print(f"Merge method: {enriched['merge_method'].iloc[0]}")

# Validate data quality
stats = validate_population_data(enriched)
print(f"Coverage: {stats['coverage_rate']:.1%}")
print(f"Mean population (populated cells): {stats['mean_pop_populated']:.1f}")

# Save enriched features
enriched.to_parquet(
    "data/processed/features/grid_features.parquet",
    compression="snappy"
)
print(f"Saved enriched features with {len(enriched)} cells")
```

---

## Testing Contract

### Unit Tests

**Test file**: `tests/unit/test_population_integrator.py`

**Required tests**:
1. `test_load_ibge_data_success`: Valid ZIP file loads correctly
2. `test_load_ibge_data_file_not_found`: Raises FileNotFoundError with remediation
3. `test_load_ibge_data_missing_pop_column`: Raises ValueError listing available columns
4. `test_load_ibge_data_crs_reprojection`: Auto-reprojects to target CRS
5. `test_merge_population_id_method`: Direct ID36 merge works correctly
6. `test_merge_population_spatial_method`: Spatial join works correctly
7. `test_merge_population_auto_fallback`: Auto method falls back to spatial if ID fails
8. `test_validate_population_data_pass`: Valid data passes all checks
9. `test_validate_population_data_fail_coverage`: Low coverage raises error
10. `test_validate_population_data_fail_total_pop`: Out-of-range total population raises error

### Integration Tests

**Test file**: `tests/integration/test_population_pipeline.py`

**Required tests**:
1. `test_end_to_end_population_integration`: Full pipeline from config to enriched features
2. `test_performance_under_10_minutes`: Entire pipeline completes within time limit
3. `test_idempotency`: Running integration twice produces identical results

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-10 | Initial contract definition for population integration feature |

---

**Approved by**: Automated planning process  
**Status**: Ready for implementation
