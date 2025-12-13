# Research: IBGE Population Data Integration

**Date**: 2025-12-10  
**Feature**: 002-population-integration  
**Status**: Complete

## Overview

This document consolidates research findings to resolve all technical unknowns identified during the planning phase. Research focused on: (1) IBGE data format and loading strategy, (2) GeoPandas best practices for spatial joins and CRS handling, (3) grid resolution change implications, (4) population feature normalization in ML pipelines.

---

## Research Task 1: IBGE Grade Estatística Data Format

**Unknown**: What is the exact structure of IBGE grade estatística 2022 ZIP files, and which column names contain population data?

### Decision: Direct ZIP Reading with Explicit Column Validation

**Rationale**: 
- IBGE distributes grade estatística data as ZIP archives containing shapefiles (`.shp`, `.dbf`, `.shx`, `.prj`)
- GeoPandas 0.14+ with PyOGRIO engine supports direct reading from ZIP without extraction: `gpd.read_file("zip://path/to/file.zip!shapefile.shp")`
- IBGE 2022 census uses standardized column names: `ID36` for cell identifier (36-character quadtree-based ID) and typically `POP` or `POPULACAO` for population count
- **Strict validation required**: Per spec FR-004, system MUST require exact `POP` column name and fail immediately if not present

**Implementation approach**:
```python
import geopandas as gpd

# Read directly from ZIP
gdf = gpd.read_file(
    "zip://data/raw/ibge_populacao_bh_grade_id36.zip",
    engine="pyogrio"  # Faster than fiona
)

# Validate required columns
if 'POP' not in gdf.columns:
    available = ', '.join(gdf.columns)
    raise ValueError(
        f"IBGE data missing required 'POP' column. "
        f"Available columns: {available}. "
        f"Remediation: Rename population column to 'POP' or update source file."
    )

if 'ID36' not in gdf.columns:
    # Use spatial join as fallback if ID36 missing
    print("Warning: ID36 column not found, will use spatial join")
```

**Alternatives considered**:
- Manual ZIP extraction → Rejected: Adds complexity, requires disk space, violates FR-004 requirement
- Flexible column name matching (regex/fuzzy) → Rejected: Spec requires strict "POP" name to enforce standardization
- Assume column positions → Rejected: Fragile, breaks with schema changes

**Performance notes**:
- PyOGRIO engine is ~2-3x faster than default Fiona for shapefile reading
- Direct ZIP reading adds negligible overhead (<1 second for typical file sizes)
- Expected file size: 5-20 MB compressed, 50-100 MB uncompressed

---

## Research Task 2: Grid Cell ID Matching Strategy

**Unknown**: Should we adopt IBGE's ID36 format during grid generation, or rely on spatial join?

### Decision: Dual Strategy - Attempt ID36 Matching, Fallback to Spatial Join

**Rationale**:
- **ID36 format**: IBGE uses a hierarchical quadtree-based 36-character string (e.g., "E2665N75155E2665N7515E2665N7515R200") encoding geographic position and resolution
- **Grid generation complexity**: Generating ID36-compatible identifiers requires implementing IBGE's quadtree algorithm, which is complex and error-prone
- **Spatial join reliability**: GeoPandas `sjoin()` with `predicate="intersects"` is robust and well-tested for polygon-polygon matching
- **Performance trade-off**: Direct ID matching is O(n), spatial join is O(n log n), but for ~15k cells the difference is <5 seconds

**Implementation approach**:
```python
# Primary method: Direct ID merge (if ID36 available in both datasets)
if 'ID36' in grid_gdf.columns and 'ID36' in ibge_gdf.columns:
    merged = grid_gdf.merge(ibge_gdf[['ID36', 'POP']], on='ID36', how='left')
    match_rate = merged['POP'].notna().sum() / len(merged)
    if match_rate < 0.95:
        print(f"Warning: Only {match_rate:.1%} cells matched by ID36")
else:
    # Fallback: Spatial join by centroid
    grid_gdf['centroid'] = grid_gdf.geometry.centroid
    merged = gpd.sjoin(
        grid_gdf,
        ibge_gdf[['geometry', 'POP']],
        how='left',
        predicate='intersects'
    )
```

**Alternatives considered**:
- ID36 generation only → Rejected: High implementation risk, not worth complexity for one-time integration
- Spatial join only → Rejected: Slightly slower, but acceptable as fallback
- Centroid-based join → Selected as fallback: Simpler than full polygon intersection

**Validation strategy**:
- After merge, verify >95% of cells have population data (SC-003)
- Log unmatched cells for manual inspection (likely edge cases or water bodies)

---

## Research Task 3: Coordinate Reference System (CRS) Handling

**Unknown**: What CRS does IBGE use, and how should we handle potential mismatches?

### Decision: Standardize to EPSG:4326 (WGS84) with Automatic Reprojection

**Rationale**:
- **IBGE standard**: IBGE grade estatística typically uses SIRGAS 2000 (EPSG:4674), which is the official Brazilian geodetic system
- **Project standard**: Current grid generation likely uses EPSG:4326 (WGS84) for lat/lon simplicity
- **Compatibility**: SIRGAS 2000 and WGS84 are very similar (<1 meter difference for Belo Horizonte), but must be handled explicitly to avoid spatial misalignment
- **GeoPandas automation**: Built-in `.to_crs()` method handles reprojection transparently

**Implementation approach**:
```python
# Load IBGE data
ibge_gdf = gpd.read_file("zip://data/raw/ibge_populacao_bh_grade_id36.zip")
print(f"IBGE CRS: {ibge_gdf.crs}")

# Load grid
grid_gdf = gpd.read_parquet("data/processed/grids/grid_200m.parquet")
print(f"Grid CRS: {grid_gdf.crs}")

# Reproject if needed
target_crs = "EPSG:4326"  # WGS84
if ibge_gdf.crs != target_crs:
    print(f"Reprojecting IBGE data from {ibge_gdf.crs} to {target_crs}")
    ibge_gdf = ibge_gdf.to_crs(target_crs)

if grid_gdf.crs != target_crs:
    print(f"Reprojecting grid from {grid_gdf.crs} to {target_crs}")
    grid_gdf = grid_gdf.to_crs(target_crs)
```

**Alternatives considered**:
- Keep original CRS, convert only for merge → Rejected: Error-prone, better to standardize early
- Use SIRGAS 2000 as standard → Rejected: WGS84 more universal, easier for international collaboration
- Manual CRS validation without auto-reproject → Rejected: Adds user burden, fails FR-006 requirement

**Performance notes**:
- Reprojection is fast: ~1-2 seconds for 15k polygons
- Accuracy: Sub-meter precision sufficient for 200m grid cells

---

## Research Task 4: Grid Resolution Change Implications

**Unknown**: How does changing from 250m to 200m affect cell count, processing time, and memory usage?

### Decision: Accept Increased Cell Count (~2x) with Performance Monitoring

**Rationale**:
- **Cell count increase**: 250m grid → ~7,000 cells (based on config), 200m grid → ~15,000-20,000 cells (2.5x-3x increase)
- **Memory impact**: Each cell stores ~10-15 features, approximately 1-2 KB per cell in memory → 200m grid requires ~30-40 MB (well within 8GB RAM constraint)
- **Processing time**: 
  - Grid generation: ~30 seconds for 250m → ~60-90 seconds for 200m (scales with cell count)
  - Transit feature extraction: ~2-3 minutes for 250m → ~5-7 minutes for 200m (spatial operations dominate)
  - Population merge: ~10-20 seconds (spatial join on 15k cells)
  - **Total pipeline**: Estimated 8-10 minutes (within SC-002 threshold of 10 minutes)
- **Model training impact**: Increased sample size generally improves model generalization, but training time may increase 10-20%

**Implementation approach**:
```python
# Add performance tracking to pipeline
import time
import logging

def track_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__name__} completed in {elapsed:.1f}s")
        return result
    return wrapper

@track_performance
def generate_grid_200m():
    # Grid generation logic
    pass

@track_performance
def extract_transit_features():
    # Feature extraction logic
    pass

@track_performance
def integrate_population():
    # Population merge logic
    pass
```

**Alternatives considered**:
- Keep 250m, downsample IBGE to match → Rejected: Loses population resolution, defeats purpose
- Test multiple resolutions (150m, 200m, 250m) → Rejected: Out of scope, 200m is optimal match to IBGE
- Implement chunked processing for memory efficiency → Deferred: Only needed if 8GB RAM exceeded (unlikely)

**Validation metrics**:
- Measure actual processing time and fail if >10 minutes (FR-017)
- Monitor memory usage, optimize if approaching limits
- Compare model performance before/after to ensure quality improvement

---

## Research Task 5: Population Feature Normalization in ML Pipeline

**Unknown**: How should population be normalized relative to transit features to ensure fair feature weighting?

### Decision: StandardScaler for All Features (Including Population)

**Rationale**:
- **Current approach**: Transit features (stop_count, route_count, daily_trips) already use StandardScaler (z-score normalization) per config
- **Population distribution**: Expected to be right-skewed (many low-pop cells, few high-pop cells), similar to transit features
- **StandardScaler benefits**: 
  - Centers features to mean=0, std=1, giving equal initial weight
  - Handles outliers better than Min-Max scaling for skewed distributions
  - Maintains interpretability (z-scores show "standard deviations from mean")
- **Consistency**: Using same normalization method for all features simplifies pipeline and prevents scale-related bugs

**Implementation approach**:
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load enriched features
df = pd.read_parquet("data/processed/features/grid_features.parquet")

# Define feature columns
transit_features = ['stop_count', 'route_count', 'daily_trips']
demographic_features = ['population']
all_features = transit_features + demographic_features

# Normalize
scaler = StandardScaler()
df[all_features] = scaler.fit_transform(df[all_features])

# Verify normalization
assert df[all_features].mean().abs().max() < 0.01, "Mean should be ~0"
assert (df[all_features].std() - 1.0).abs().max() < 0.01, "Std should be ~1"
```

**Alternatives considered**:
- Min-Max scaling [0, 1] → Rejected: Sensitive to outliers, loses information about distribution shape
- Log transformation + StandardScaler → Deferred: May improve if population distribution extremely skewed, test during model training
- Separate scaling per feature type → Rejected: Unnecessary complexity, StandardScaler handles mixed distributions well

**Feature engineering considerations**:
- Population used directly (not density) because cell size is constant (200m²)
- If future work adds variable cell sizes, switch to population density (pop/km²)
- Feature importance analysis will reveal if population needs transformation

---

## Research Task 6: Zero-Population Cell Handling

**Unknown**: How should cells with zero population (parks, water, industrial) be handled in statistics and model training?

### Decision: Preserve Zeros, Exclude from Mean, Include in Model Training

**Rationale**:
- **Preserve zeros**: Zero is meaningful data (uninhabited areas), not missing data. Setting to NaN or imputing would lose information.
- **Statistical reporting**: 
  - Calculate mean ONLY from populated cells (pop > 0) for interpretability (FR-012)
  - Report zero-cell count separately to understand uninhabited area coverage
  - Example: "Mean population: 150 per cell (12,000 populated cells), 2,000 zero-pop cells (parks/water)"
- **Model training**: Include zero-population cells with label=underserved (low transit demand + low transit supply = appropriately served)
- **Edge case validation**: Zero-pop cells should correlate with expected land use (SC-005: <10% of total cells)

**Implementation approach**:
```python
import pandas as pd

# Load enriched data
df = pd.read_parquet("data/processed/features/grid_features.parquet")

# Calculate statistics
total_cells = len(df)
zero_pop_cells = (df['population'] == 0).sum()
populated_cells = total_cells - zero_pop_cells

stats = {
    'total_population': df['population'].sum(),
    'total_cells': total_cells,
    'populated_cells': populated_cells,
    'zero_pop_cells': zero_pop_cells,
    'zero_pop_pct': zero_pop_cells / total_cells,
    'mean_pop_all': df['population'].mean(),
    'mean_pop_populated': df[df['population'] > 0]['population'].mean(),
    'min_pop': df['population'].min(),
    'max_pop': df['population'].max(),
}

print(f"Population Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.2f}")

# Validate zero-pop percentage
if stats['zero_pop_pct'] > 0.10:
    print(f"WARNING: {stats['zero_pop_pct']:.1%} cells have zero population (expected <10%)")
```

**Alternatives considered**:
- Impute zeros with neighborhood average → Rejected: Loses information, creates artificial data
- Exclude zero-pop cells from model training → Rejected: These cells are valid data points for "appropriately underserved"
- Flag zeros as missing data → Rejected: Zero is real data, not missing

**Validation checks**:
- Verify zero-pop cells spatial distribution (should cluster in parks, rivers, industrial zones)
- Cross-reference with OpenStreetMap land use data if validation needed

---

## Research Task 7: IBGE File Error Handling Strategy

**Unknown**: How should the pipeline handle various IBGE file read failures (missing, corrupted, wrong format)?

### Decision: Fail-Fast with Actionable Error Messages

**Rationale**:
- **Per FR-004**: System MUST fail immediately on file read errors with clear remediation steps
- **Error categories**:
  1. File not found → Missing file, wrong path, or incorrect filename
  2. Corrupted ZIP → Incomplete download, disk corruption
  3. Invalid shapefile → Wrong file type, missing components (.shp/.dbf/.shx)
  4. Schema mismatch → Missing required columns (POP, geometry)
- **User experience**: Clear error messages save debugging time and prevent silent failures

**Implementation approach**:
```python
import os
from pathlib import Path
import geopandas as gpd

def load_ibge_data(zip_path: str) -> gpd.GeoDataFrame:
    """
    Load IBGE population data with comprehensive error handling.
    
    Args:
        zip_path: Path to IBGE ZIP file
        
    Returns:
        GeoDataFrame with population data
        
    Raises:
        FileNotFoundError: If ZIP file doesn't exist
        ValueError: If file format or schema is invalid
    """
    # Check file exists
    if not Path(zip_path).exists():
        raise FileNotFoundError(
            f"IBGE data file not found: {zip_path}\n"
            f"Remediation steps:\n"
            f"1. Download from IBGE website: https://www.ibge.gov.br/geociencias/...\n"
            f"2. Save to: data/raw/ibge_populacao_bh_grade_id36.zip\n"
            f"3. Verify file integrity with checksum (see documentation)"
        )
    
    # Attempt to read ZIP
    try:
        gdf = gpd.read_file(f"zip://{zip_path}", engine="pyogrio")
    except Exception as e:
        raise ValueError(
            f"Failed to read IBGE ZIP file: {zip_path}\n"
            f"Error: {str(e)}\n"
            f"Possible causes:\n"
            f"- Corrupted ZIP archive (re-download file)\n"
            f"- Invalid shapefile format (verify file contents)\n"
            f"- Missing shapefile components (.shp/.dbf/.shx/.prj)\n"
            f"Remediation: Download fresh copy from IBGE"
        )
    
    # Validate required columns
    if 'POP' not in gdf.columns:
        available = ', '.join(gdf.columns)
        raise ValueError(
            f"IBGE data missing required 'POP' column.\n"
            f"Available columns: {available}\n"
            f"Remediation: Rename population column to 'POP' or update source file"
        )
    
    # Validate geometry
    if gdf.geometry.isna().any():
        raise ValueError("IBGE data contains invalid geometries")
    
    return gdf
```

**Alternatives considered**:
- Silent failure with warning → Rejected: Violates FR-004, wastes computation time
- Retry logic with backoff → Rejected: File errors are deterministic, retries won't help
- Fallback to alternative data source → Rejected: Out of scope, no alternative source specified

**Testing strategy**:
- Unit test with missing file path
- Unit test with corrupted ZIP (simulate with invalid bytes)
- Unit test with valid ZIP but wrong schema
- Integration test with real IBGE file (happy path)

---

## Summary of Key Decisions

| Research Area | Decision | Rationale |
|---------------|----------|-----------|
| IBGE Data Loading | Direct ZIP read with strict `POP` column validation | Efficient, enforces standardization per FR-004 |
| Cell ID Matching | Dual strategy: ID36 merge → spatial join fallback | Balance simplicity and reliability |
| CRS Handling | Auto-reproject to EPSG:4326 (WGS84) | Standardization, international compatibility |
| Grid Resolution | Accept 200m (~15-20k cells) with perf monitoring | Matches IBGE, within memory/time constraints |
| Normalization | StandardScaler for all features | Consistency, handles skewed distributions |
| Zero Population | Preserve zeros, exclude from mean, include in training | Zero is meaningful data, not missing |
| Error Handling | Fail-fast with actionable messages | Saves debugging time, prevents silent failures |

---

## Open Questions for Implementation Phase

1. **Baseline comparison**: Should we preserve 250m grid results for model performance comparison? **Recommendation**: Yes, archive current best_model.onnx and metrics before regeneration.

2. **Population feature weight**: Should population have a configurable weight in composite scoring (like transit features)? **Recommendation**: No, use StandardScaler and let model learn weights automatically via feature importance.

3. **Spatial visualization**: Should we generate maps showing population distribution vs transit coverage? **Recommendation**: Defer to future work, not in current spec scope.

4. **Performance optimization**: If 10-minute threshold exceeded, should we implement Dask for parallel processing? **Recommendation**: Measure first, optimize only if needed.

---

**Next Steps**: Proceed to Phase 1 (Design) to generate data models and contracts based on these research findings.
