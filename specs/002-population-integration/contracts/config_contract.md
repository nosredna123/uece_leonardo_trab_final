# Configuration Contract: model_config.yaml Updates

**Configuration File**: `config/model_config.yaml`  
**Purpose**: Define grid resolution and population integration settings  
**Created**: 2025-12-10  
**Feature**: 002-population-integration

---

## Required Configuration Changes

### Section: `grid`

**Changes**:
```yaml
grid:
  # CHANGED: Reduced from 250m to 200m to match IBGE resolution
  cell_size_meters: 200
  
  # UNCHANGED: Geographic bounds (existing)
  bounds:
    lat_min: -20.046411
    lat_max: -19.758246
    lon_min: -44.081380
    lon_max: -43.843522
  
  # UPDATED: Expected cell count increased due to smaller cells
  expected_cells: 17000  # Was: 13000
  
  # UPDATED: Expected area per cell reduced
  expected_area_km2: 0.04  # Was: 0.0625 (200m × 200m vs 250m × 250m)
```

**Rationale**: Grid resolution must match IBGE data (200m) to enable direct cell-to-cell mapping without spatial aggregation complexity.

**Validation**: Grid generator will verify actual cell count is within ±20% of expected_cells.

---

### Section: `features` (NEW subsection)

**Addition**:
```yaml
features:
  # Existing weights for composite transit score (unchanged)
  stop_count_weight: 0.4
  route_count_weight: 0.3
  daily_trips_weight: 0.3
  
  normalization_method: "StandardScaler"
  include_density: true
  include_diversity: true
  
  # NEW: Population integration settings
  population:
    # Path to IBGE ZIP file (relative to project root)
    source_file: "data/raw/ibge_populacao_bh_grade_id36.zip"
    
    # Required column name in IBGE data (strict match)
    required_column: "POP"
    
    # Population value validation range
    min_expected_population: 0
    max_expected_population: 3000  # Per 200m cell, flag if exceeded
    
    # Total population range for Belo Horizonte (validation)
    total_population_range:
      min: 2000000  # 2.0 million
      max: 2800000  # 2.8 million
    
    # Merge quality thresholds
    coverage_threshold: 0.95  # Minimum 95% cells must have population data (SC-003)
    max_zero_pop_pct: 0.10    # Maximum 10% cells can have zero population (SC-005)
    
    # CRS handling
    target_crs: "EPSG:4326"  # WGS84, must match grid CRS
```

**Purpose**: Centralize all population integration parameters for easy configuration and validation.

**Usage**: `population_integrator.py` reads these settings to validate data and apply thresholds.

---

### Section: `training` (UPDATE subsection)

**Addition**:
```yaml
training:
  random_seed: 42
  test_size: 0.15
  val_size: 0.15
  cv_folds: 5
  stratify: true
  n_jobs: -1
  
  # NEW: Explicit feature set definition
  feature_columns:
    # Transit features (existing)
    - stop_count
    - route_count
    - daily_trips
    # Optional derived features
    - stop_density
    - route_diversity
    # NEW: Demographic feature
    - population
```

**Purpose**: Explicitly list features used in model training, ensuring population is included.

**Usage**: `train.py` reads `feature_columns` to select and normalize features before model training.

---

### Section: `performance` (NEW)

**Addition**:
```yaml
# NEW: Performance monitoring and thresholds
performance:
  # Maximum allowed pipeline execution time (FR-017, SC-002)
  max_total_time_minutes: 10
  max_population_integration_minutes: 5
  
  # Memory usage monitoring (optional)
  max_memory_gb: 8
  
  # Fail-fast behavior
  fail_on_timeout: true  # Terminate if time thresholds exceeded
  
  # Logging level for performance metrics
  log_level: "INFO"
```

**Purpose**: Enforce performance requirements (SC-002) and enable fail-fast behavior per FR-017.

**Usage**: Pipeline scripts measure execution time and compare against thresholds.

---

## Configuration Validation

### Pre-Execution Checks

Before running the pipeline, validate:

1. **Grid configuration**:
   - `cell_size_meters` is exactly 200 (strict equality)
   - `bounds` define valid lat/lon ranges
   - `expected_cells` is reasonable (10,000 - 25,000)

2. **Population configuration**:
   - `source_file` path exists and is readable
   - `required_column` is non-empty string
   - `coverage_threshold` in range [0.90, 1.00]
   - `max_zero_pop_pct` in range [0.00, 0.20]

3. **Feature configuration**:
   - `feature_columns` includes "population"
   - All feature columns exist in enriched dataset
   - No duplicate column names

**Validation script**: `src/data/validate_config.py` (to be implemented)

---

## Backward Compatibility

### Breaking Changes

⚠️ **Warning**: Updating `cell_size_meters` from 250 to 200 is a **breaking change** that requires:
1. Regenerating the entire grid
2. Re-extracting all transit features
3. Retraining all models

### Migration Strategy

**Option 1: Clean regeneration (recommended)**:
```bash
# Backup existing results
mv models/transit_coverage models/transit_coverage_250m_backup
mv data/processed/grids data/processed/grids_250m_backup

# Update config
# Edit config/model_config.yaml: cell_size_meters = 200

# Regenerate pipeline
bash run_pipeline.sh
```

**Option 2: Parallel configurations**:
Create separate config files for comparison:
- `config/model_config_250m.yaml` (baseline)
- `config/model_config_200m.yaml` (with population)

Allow pipeline to accept config path as argument.

---

## Usage Example

```python
import yaml
from pathlib import Path

# Load configuration
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access population settings
pop_config = config["features"]["population"]
source_file = pop_config["source_file"]
coverage_threshold = pop_config["coverage_threshold"]

print(f"Population source: {source_file}")
print(f"Required coverage: {coverage_threshold:.1%}")

# Access grid settings
grid_size = config["grid"]["cell_size_meters"]
expected_cells = config["grid"]["expected_cells"]

print(f"Grid resolution: {grid_size}m")
print(f"Expected cells: {expected_cells:,}")

# Access feature columns
feature_cols = config["training"]["feature_columns"]
print(f"Training features: {', '.join(feature_cols)}")
```

---

## Environment Variables (Alternative)

For sensitive or environment-specific paths, consider environment variable overrides:

```bash
# Override IBGE file path
export IBGE_POPULATION_FILE="/path/to/custom/ibge_data.zip"

# Override performance thresholds
export MAX_PIPELINE_TIME_MINUTES=15
```

**Configuration loading with overrides**:
```python
import os
import yaml

with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Override from environment if set
source_file = os.getenv(
    "IBGE_POPULATION_FILE",
    config["features"]["population"]["source_file"]
)

max_time = int(os.getenv(
    "MAX_PIPELINE_TIME_MINUTES",
    config["performance"]["max_total_time_minutes"]
))
```

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-10 | Initial configuration contract for 200m grid and population integration |

---

**Status**: Ready for implementation
