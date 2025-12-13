# Feature Specification: IBGE Population Data Integration

**Feature Branch**: `002-population-integration`  
**Created**: 2025-12-10  
**Status**: Draft  
**Input**: User description: "Integrate IBGE population data to improve transit coverage model accuracy"

## Clarifications

### Session 2025-12-10

- Q: How should the pipeline handle IBGE file read failures (corrupted, network error, wrong format)? → A: Fail fast with clear error - Stop immediately, display specific error message with remediation steps
- Q: How should the system identify which column contains population data in the IBGE file? → A: Strict exact match - Require exactly `POP` column name; fail if not present (forces standardization)
- Q: What should happen if grid regeneration and population integration exceeds the 10-minute performance threshold? → A: Fail immediately - Terminate with error suggesting data size reduction or hardware upgrade
- Q: How should population be allocated when IBGE 200m cells partially overlap multiple 250m grid cells? → A: Assign to largest overlap - Give entire population to the 250m cell with the largest intersection area (simpler, faster)
- Q: How should zero-population cells be handled when calculating summary statistics like mean population per cell? → A: Exclude from mean - Calculate mean only from cells with pop > 0; report zero-cell count separately (more meaningful for urban analysis)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Data Scientist Enriches Model with Population Context (Priority: P1)

A data scientist wants to improve the transit coverage classifier by incorporating population density as an additional feature. The current model relies solely on transit metrics (stops, routes, trips) but doesn't account for population distribution, which can lead to identifying densely populated areas with moderate transit as underserved, or sparsely populated areas with low transit as well-served.

**Why this priority**: This addresses the core model limitation identified in the problem statement - the "almost perfect models" issue. Population data provides crucial context to distinguish between genuinely underserved areas and low-population zones that naturally have less transit infrastructure.

**Independent Test**: Can be fully tested by regenerating the grid at 200m resolution to match IBGE data, extracting transit features for the 200m grid, loading IBGE population data with matching cell IDs, merging by cell ID, and verifying the enriched dataset contains accurate population counts.

**Acceptance Scenarios**:

1. **Given** the configuration is updated to use 200m grid cells, **When** the grid generation pipeline is executed, **Then** the system creates a 200m grid matching IBGE's resolution and coordinate system
2. **Given** the IBGE grade estatística 2022 ZIP file in `data/raw/`, **When** the data scientist runs the population extraction pipeline, **Then** the system loads 200m grid cells with ID36 identifiers and population counts
3. **Given** transit features extracted for the 200m grid and IBGE population data, **When** merged by matching cell identifiers, **Then** the enriched dataset contains a new `population` column with non-negative integer values
4. **Given** the enriched grid features, **When** saved to parquet format, **Then** the file is stored at `data/processed/features/grid_features.parquet` with the population column included

---

### User Story 2 - Model Trainer Validates Population Feature Impact (Priority: P2)

A model trainer wants to verify that adding population data improves model performance and reduces the "almost perfect" overfitting issue. They need to retrain the model with the new population feature and compare metrics against the baseline.

**Why this priority**: Validating feature effectiveness ensures the integration effort delivers measurable value. This prevents adding unnecessary complexity without performance gains.

**Independent Test**: Can be tested by training both baseline (without population) and enriched (with population) models, comparing validation metrics (accuracy, F1-score, AUC-ROC), and verifying that the enriched model shows improved generalization on unseen data.

**Acceptance Scenarios**:

1. **Given** enriched grid features with population data, **When** the model training pipeline is executed, **Then** the population feature is included in the feature set and properly normalized
2. **Given** trained models with and without population features, **When** validation metrics are compared, **Then** the enriched model demonstrates improved performance or maintained performance with better interpretability
3. **Given** feature importance analysis, **When** examining the trained model, **Then** population appears as a meaningful feature with non-zero importance weight

---

### User Story 3 - Urban Planner Interprets Population-Aware Classifications (Priority: P3)

An urban planner wants to understand how population density influences transit coverage classifications. High-population areas classified as underserved should be prioritized differently than low-population areas with the same transit metrics.

**Why this priority**: This enables better decision-making by providing context-aware classifications that account for demand (population) versus supply (transit infrastructure).

**Independent Test**: Can be tested by querying the API or notebook with sample locations of varying population densities, observing how classifications differ between high-pop and low-pop areas with similar transit metrics, and verifying that explanations include population context.

**Acceptance Scenarios**:

1. **Given** two grid cells with identical transit metrics but different population densities, **When** classified by the enriched model, **Then** the high-population cell is more likely to be classified as underserved than the low-population cell
2. **Given** a classification result from the API, **When** requesting feature contributions, **Then** the response includes population as one of the input features used for the prediction

---

### User Story 4 - Researcher Reviews Population Impact in Technical Report (Priority: P2)

A researcher or evaluator wants to understand how the population feature contributes to the model and why it was included, documented in the technical report with clear justification and analysis.

**Why this priority**: Academic and technical documentation must explain design decisions and feature engineering choices. This ensures reproducibility, enables peer review, and demonstrates the scientific rigor of the model development process.

**Independent Test**: Can be tested by reviewing the generated technical report (`reports/relatorio_tecnico.md` and PDF) to verify it contains dedicated sections explaining: (1) rationale for adding population as an independent feature, (2) analysis of population's contribution to model predictions via feature importance, (3) comparison of model performance with and without population data.

**Acceptance Scenarios**:

1. **Given** the technical report generation script, **When** executed after model training with population features, **Then** the report includes a section titled "Population Data Integration" explaining why population is used as an independent feature
2. **Given** the report section on population integration, **When** reviewed, **Then** it clearly articulates the problem being solved (distinguishing genuinely underserved high-population areas from appropriately served low-population areas)
3. **Given** the feature importance analysis in the report, **When** examining the results, **Then** population's contribution percentage and ranking among all features is displayed
4. **Given** the model comparison section, **When** comparing baseline (transit-only) vs enriched (transit + population) models, **Then** the report shows performance metrics side-by-side with interpretation of improvements or changes

---

### Edge Cases

- **Empty population cells**: What happens when a 200m grid cell has zero population (e.g., parks, water bodies, industrial zones)? System should handle zero values without errors and these cells should be classified appropriately as low-priority for transit expansion.
- **Missing IBGE data**: What if certain 200m cells in the IBGE dataset have missing population values? System should document missing data handling strategy (e.g., impute with 0 or flag for review).
- **IBGE file read failures**: If the IBGE ZIP file cannot be read (corrupted file, network error, wrong format, missing file), the pipeline must fail immediately with a clear error message. The error message should specify: (1) the exact file path expected, (2) the specific failure reason (e.g., "file not found", "corrupted ZIP archive", "invalid shapefile format"), and (3) remediation steps (e.g., "download file from IBGE website to data/raw/", "verify file integrity with checksum"). This prevents silent failures and wasted computation.
- **Unmatched cell IDs**: What if the generated 200m grid uses different cell IDs than IBGE's ID36 format? System should either adopt ID36 format during grid generation or create a spatial join as fallback if direct ID matching fails.
- **Buffer zone edges**: What happens to grid cells at the edge of the 2km buffer that fall outside IBGE data coverage? System should flag these cells and optionally exclude them from model training.
- **Coordinate system mismatch**: What if IBGE data uses a different CRS than the generated grid? System must detect CRS differences and reproject one dataset to match the other before merging.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST update `config/model_config.yaml` to change grid cell size from 250m to 200m to match IBGE resolution
- **FR-002**: System MUST regenerate the geographic grid at 200m resolution using the updated configuration, creating cells that align with IBGE's ID36 grade system
- **FR-003**: System MUST re-extract transit features (stop counts, route counts, daily trips) for the new 200m grid cells
- **FR-004**: System MUST load IBGE grade estatística 2022 data directly from the ZIP file at `data/raw/ibge_populacao_bh_grade_id36.zip` without requiring manual extraction. System MUST require an exact column named `POP` for population values; if not present, terminate immediately with error listing available column names. If the file cannot be read (missing, corrupted, wrong format, network error), the system MUST immediately terminate with a clear error message specifying: (1) the expected file path, (2) the specific failure reason, and (3) remediation steps. No silent failures or fallback behavior is permitted.
- **FR-005**: System MUST filter IBGE 200m grid cells to cover Belo Horizonte geographic bounds plus a 2km buffer zone to ensure complete coverage
- **FR-006**: System MUST validate that both IBGE data and generated grid use compatible coordinate reference systems (CRS), and reproject if necessary to EPSG:4326 (WGS84)
- **FR-007**: System MUST merge IBGE population data with transit features by matching cell identifiers (either using ID36 format or spatial join as fallback)
- **FR-008**: System MUST handle zero-population cells by preserving the zero value rather than treating it as missing data
- **FR-009**: System MUST add a new column named `population` containing non-negative integer values representing the total population in each 200m cell
- **FR-010**: System MUST preserve all existing transit feature columns when adding the population column
- **FR-011**: System MUST save the enriched grid features to `data/processed/features/grid_features.parquet` in Parquet format with appropriate data types
- **FR-012**: System MUST log summary statistics including: total population loaded, total cell count, number of cells with zero population, number of populated cells (pop > 0), min/max population per cell, and mean population calculated ONLY from populated cells (excluding zeros). Cells with missing data must be reported separately and total cell count comparison documented.
- **FR-013**: System MUST create a Python module `src/data/population_integrator.py` that encapsulates the population loading and merging logic as reusable functions
- **FR-014**: System MUST update the feature extraction pipeline to include population as a normalized feature when training models
- **FR-015**: System MUST update the technical report generation script (`generate_report.py`) to include a dedicated section explaining population data integration, its rationale, and impact on model performance
- **FR-016**: System MUST update `README.md` to document the population integration feature, including data source (IBGE Censo 2022), why it addresses the model limitation, and how to regenerate results with the 200m grid
- **FR-017**: System MUST enforce the 5-minute performance threshold for population aggregation (SC-002). If processing exceeds this limit, the system MUST immediately terminate with an error message suggesting: (1) data size reduction strategies (e.g., smaller geographic bounds), (2) hardware upgrade recommendations (RAM, CPU cores), or (3) enabling chunked processing mode.

### Key Entities *(include if feature involves data)*

- **IBGE Grade Estatística Cell (200m)**: Represents a 200-meter resolution grid cell from IBGE's 2022 census, containing attributes: cell identifier (ID36 format), geometry (polygon), population count (integer), and geographic bounds. Each cell covers approximately 0.04 km² (200m × 200m).
  
- **Transit Coverage Grid Cell (200m)**: Represents the regenerated 200-meter resolution analysis grid, containing attributes: cell_id (unique identifier, preferably matching ID36 format), geometry (polygon), transit metrics (stop_count, route_count, daily_trips), normalized features, and geographic bounds (lat/lon). This grid now matches IBGE's resolution exactly.

- **Enriched Grid Features**: The merged dataset combining transit metrics and population data for each 200m cell. Contains all original transit features plus the new population field, enabling multi-dimensional analysis of transit coverage relative to population demand. Cells are matched by ID or spatial location.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Population data is successfully loaded from the IBGE ZIP file with at least 90% of expected cells for Belo Horizonte's geographic extent (approximately 15,000-20,000 200m cells based on area calculation)
- **SC-002**: Grid regeneration and population integration completes in under 10 minutes on standard hardware (e.g., laptop with 16GB RAM, 4-core CPU) for the full Belo Horizonte dataset. If exceeded, system fails per FR-017.
- **SC-003**: The enriched grid features file contains population values for at least 95% of the 200m grid cells (approximately 15,000-20,000 cells matching IBGE coverage)
- **SC-004**: Population distribution shows realistic patterns: mean population (calculated from populated cells only, excluding zeros) between 60-400 per 200m cell, with higher densities in central urban areas and lower densities in peripheral zones
- **SC-005**: Zero population cells represent less than 10% of total cells and correlate with expected non-residential zones (e.g., large parks, water bodies, industrial areas)
- **SC-006**: Model retraining with population feature completes successfully without errors, and the feature appears in the trained model's feature set
- **SC-007**: Feature importance analysis shows population contributes at least 5% to model predictions, indicating it provides meaningful signal
- **SC-008**: The enriched dataset increases in file size by no more than 15% compared to the baseline grid features file (efficient storage)
- **SC-009**: Data pipeline documentation is updated with clear instructions for reproducing the population integration process
- **SC-010**: Validation checks confirm no data corruption: total population across all cells matches expected Belo Horizonte population (approximately 2.3-2.5 million based on census data)

## Assumptions *(generated during specification)*

1. **Data availability**: The IBGE file `ibge_populacao_bh_grade_id36.zip` contains complete and accurate 2022 census population data for Belo Horizonte at 200m resolution using the ID36 grade reference system.

2. **Grid regeneration feasibility**: Regenerating the grid at 200m resolution and re-extracting all transit features is acceptable within the project timeline and does not invalidate prior work (existing 250m results can be retained for comparison).

3. **Computational resources**: The development environment has sufficient memory (minimum 8GB RAM) and GeoPandas capabilities to handle the 200m grid (approximately 15,000-20,000 cells, more than the previous 250m grid).

4. **Population distribution**: Census 2022 population counts are reported as totals per 200m cell without further demographic breakdown. Age, income, or other demographic variables are not required for this initial integration.

5. **Static population data**: Population values represent a snapshot from 2022 census and are treated as static. Real-time or projected population changes are out of scope.

6. **Buffer zone rationale**: The 2km buffer around Belo Horizonte ensures complete coverage of transit infrastructure that may extend beyond strict city boundaries, capturing suburban transit stops and edge-case scenarios.

7. **Cell ID matching**: IBGE's ID36 cell identifiers can be adopted during grid generation, or a reliable spatial join can match cells by centroid/geometry if direct ID matching is not feasible.

8. **Model integration**: The existing model training pipeline (`src/models/train.py`) can accommodate the new 200m grid and additional population feature without requiring architectural changes to the ML algorithms or pipeline structure.

9. **Feature normalization**: Population will use the same StandardScaler normalization as existing transit features to ensure consistent scale and prevent any single feature from dominating the model.

10. **Performance baseline**: Current model performance metrics (accuracy, precision, recall, F1-score) are documented and available for comparison after population integration to measure impact.

## Dependencies *(generated during specification)*

### Data Dependencies
- **IBGE Grade Estatística 2022**: Raw population data file must be present at `data/raw/ibge_populacao_bh_grade_id36.zip`
- **GTFS data**: BHTrans GTFS data must be available to re-extract transit features for the new 200m grid
- **Configuration update**: `config/model_config.yaml` must be updated with 200m grid parameters before regeneration

### System Dependencies
- **GeoPandas**: Required for loading shapefiles from ZIP, spatial operations (intersection, sjoin), and CRS transformations
- **PyOGRIO engine**: Preferred engine for efficient reading of geospatial data directly from compressed archives
- **Pandas**: For dataframe operations, aggregation (groupby, sum), and merging datasets
- **PyArrow or fastparquet**: For reading and writing Parquet files efficiently

### Process Dependencies
- **Configuration update**: `config/model_config.yaml` must be updated first to change cell_size_meters from 250 to 200
- **Grid regeneration**: New 200m grid must be generated using updated configuration
- **Feature re-extraction**: Transit features must be re-extracted for the new 200m grid before population merge
- **Population integration**: IBGE data loading and merging occurs after transit feature extraction is complete

### External Dependencies
- **Geographic bounds**: Belo Horizonte bounding box coordinates must be accurately defined in configuration to filter IBGE data appropriately
- **CRS definitions**: Both datasets must have well-defined CRS metadata to enable accurate spatial operations

## Out of Scope *(generated during specification)*

- **Demographic segmentation**: Age groups, income levels, ethnicity, or other demographic breakdowns beyond total population count
- **Temporal analysis**: Historical population trends, projections, or time-series analysis of population changes
- **Administrative boundaries**: Integration of neighborhood, district, or census tract boundaries for region-specific analysis
- **Real-time data**: Live population estimates or mobility patterns from mobile data or other dynamic sources
- **Other cities**: Population integration for cities other than Belo Horizonte
- **Alternative resolution grids**: Testing different grid cell sizes (e.g., 100m, 500m) beyond the chosen 200m resolution
- **Retention of 250m results**: Maintaining or comparing the old 250m grid results (this is optional; the 200m grid replaces it)
- **Data quality assessment**: Validation of IBGE data accuracy or comparison with alternative population sources
- **Visualization tools**: Maps or dashboards showing population distribution overlaid with transit coverage (may be added in future features)
- **API updates**: Modifying the inference API to accept or return population-related parameters (API remains unchanged, only model internals are enhanced)

## Technical Notes *(for planning phase)*

### Data Format Details
- **IBGE file structure**: ZIP archive likely contains one or more shapefiles (`.shp`, `.shx`, `.dbf`, `.prj`) with polygon geometries and population attributes
- **Expected IBGE columns**: Geometry column plus attributes. System MUST require an exact column named `POP` for population count; if not present, fail immediately with error listing available columns. Cell identifier expected as `ID36`. Any other column names will cause failure to enforce standardization.
- **File size**: IBGE ZIP file size should be verified; large files (>100MB) may require streaming or chunked processing

### Merging Strategy
- **Primary method**: Direct merge on cell ID if grid generation adopts IBGE's ID36 format
- **Fallback method**: Spatial join using `gpd.sjoin()` with `predicate="intersects"` or matching by centroid if cell IDs differ
- **Validation**: After merge, verify that at least 95% of cells have population data; investigate unmatched cells

### Pipeline Integration Points
- **Configuration update**: Modify `config/model_config.yaml` to set `cell_size_meters: 200`
- **Grid regeneration**: Run `src/data/grid_generator.py` with updated config to create 200m grid
- **Feature re-extraction**: Run `src/data/feature_extractor.py` to extract transit metrics for 200m grid
- **New module**: `src/data/population_integrator.py` will contain `load_ibge_data()`, `filter_to_bounds()`, and `merge_population()` functions
- **Pipeline script**: Update `run_pipeline.sh` to include all steps in sequence: config → grid → features → population → labels → training
- **Notebook support**: Update `02_feature_engineering.ipynb` to demonstrate the 200m grid and population integration with visualizations

### Configuration Changes
```yaml
# Update in config/model_config.yaml
grid:
  cell_size_meters: 200  # CHANGED from 250 to match IBGE resolution
  # ... rest of grid config remains the same ...

features:
  # ... existing weights ...
  population_weight: 0.0  # Set to 0 initially, can be tuned if using weighted composite scoring
  
  # Population-specific settings
  population:
    source_file: "data/raw/ibge_populacao_bh_grade_id36.zip"
    min_expected_population: 0
    max_expected_population: 3000  # Flag cells exceeding this for review (lower threshold for smaller cells)
    normalization_method: "StandardScaler"
```

### Validation Checks
- **Total population sanity check**: Sum of all cell populations should approximate Belo Horizonte's census total (±10% tolerance)
- **Spatial coverage**: Verify that grid cells outside the IBGE data extent are flagged and handled appropriately
- **Missing data report**: Log any grid cells that fail to receive population data due to gaps in IBGE coverage or merge failures
- **Cell count comparison**: Compare number of generated grid cells vs IBGE cells to ensure reasonable overlap (should be >95%)

### Performance Considerations
- **Memory optimization**: For large datasets, consider processing in chunks or using Dask for out-of-core computation
- **I/O efficiency**: Read directly from ZIP using `gpd.read_file("zip://...")` to avoid extraction overhead
- **Caching strategy**: Save intermediate results (filtered IBGE data, generated grid) to avoid reprocessing during development/debugging
- **Grid regeneration impact**: Re-extracting transit features for 200m grid may take longer than the original 250m grid due to increased cell count; monitor performance and optimize if needed

### Technical Report Requirements

The technical report (`reports/relatorio_tecnico.md` and PDF) MUST include a dedicated section documenting the population data integration. This section should be structured as follows:

**Section: Population Data Integration**

1. **Motivation and Problem Statement**
   - Explain the "almost perfect models" limitation with transit-only features
   - Describe the inability to distinguish high-population underserved areas from low-population appropriately-served areas
   - Articulate why population density is a critical contextual feature for urban planning decisions

2. **Data Source and Methodology**
   - Document IBGE Censo 2022 as the population data source
   - Explain the 200m grid resolution choice (matching IBGE data to eliminate spatial aggregation complexity)
   - Describe the integration approach: direct cell ID merge vs spatial join
   - Note the grid regeneration requirement and its implications

3. **Population as an Independent Feature**
   - **Why independent?** Population represents demand for transit services, while transit metrics (stops, routes, trips) represent supply. These are fundamentally different dimensions that should be analyzed separately
   - **Normalization:** Population is normalized using StandardScaler alongside other features to ensure equal weighting opportunity in the model
   - **Feature engineering decision:** Population is used directly as a feature rather than derived metrics (e.g., population density per km²) to preserve interpretability

4. **Feature Importance Analysis**
   - Display population's contribution percentage to model predictions from feature importance analysis
   - Rank population among all features (transit and demographic)
   - Interpret whether population is a primary, secondary, or minor contributor
   - Include visualization: feature importance bar chart with population highlighted

5. **Model Performance Comparison**
   - **Baseline model:** Transit-only features (stops, routes, trips) - 250m grid
   - **Enriched model:** Transit + population features - 200m grid
   - Side-by-side comparison table:
     - Accuracy, Precision, Recall, F1-score, AUC-ROC
     - Number of features, grid resolution, total cells
   - Interpretation: Did population improve performance? If not, did it improve interpretability or reduce overfitting?

6. **Case Study Examples**
   - Provide 2-3 example grid cells showing how population context changes classification:
     - Example 1: High population + moderate transit → underserved (baseline may have missed this)
     - Example 2: Low population + low transit → appropriately served (baseline may have misclassified as underserved)
   - Include cell IDs, coordinates, actual feature values, and model predictions

7. **Limitations and Future Work**
   - Acknowledge limitations: static 2022 population, no demographic segmentation (age, income)
   - Suggest future enhancements: temporal population changes, daytime vs nighttime population, socioeconomic variables

**Report Generation Script Updates:**
- `generate_report.py` must be updated to automatically populate this section
- Extract feature importance from trained model
- Load baseline and enriched model metrics for comparison
- Generate required visualizations (feature importance chart, comparison table)
- Format section in both Markdown and PDF outputs



