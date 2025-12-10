# Feature Specification: Transit Coverage Classifier

**Feature ID:** 1-transit-coverage-classifier  
**Created:** 2025-12-09  
**Status:** Draft

---

## Overview

### Purpose

Develop a binary classification model to identify whether a given region in Belo Horizonte is **well-served** or **underserved** by public transportation, based on metrics extracted from GTFS data provided by BHTrans.

### Business Value

This classification supports the diagnosis of inequality in urban transportation coverage and provides a starting point for prioritizing public investments and improving mobility policies. The model will enable data-driven decisions for urban planning and inclusive mobility strategies.

### Scope

**In Scope:**
- Geographic grid-based analysis of Belo Horizonte
- Binary classification (well-served vs underserved regions)
- Feature extraction from GTFS data (stops, routes, trips, schedules)
- Model training using scikit-learn algorithms (Random Forest, Logistic Regression, Gradient Boosting)
- Model export to ONNX format
- API endpoint for inference

**Out of Scope:**
- Real-time transit tracking
- Multi-class classification (beyond binary)
- Integration with external demographic databases (optional for future)
- Temporal analysis across different time periods
- Other cities beyond Belo Horizonte

---

## User Scenarios & Testing

### Primary User Scenarios

**Scenario 1: Urban Planner Analyzing Coverage**
1. Urban planner accesses the API with geographic coordinates
2. System returns classification (well-served/underserved) with confidence score
3. Planner visualizes results on a map to identify underserved areas
4. Results inform decisions on where to expand transit services

**Scenario 2: Policy Maker Evaluating Equity**
1. Policy maker requests classification for all city regions
2. System processes batch requests and returns results
3. Policy maker analyzes distribution of well-served vs underserved areas
4. Findings support budget allocation and policy priorities

**Scenario 3: Researcher Validating Model**
1. Researcher loads test dataset through notebook
2. Model predicts classifications for validation set
3. Researcher reviews metrics (accuracy, precision, recall, F1-score)
4. Results are compared against ground truth labels

### Acceptance Scenarios

**Given** a geographic grid cell with calculated transit metrics  
**When** the model receives the features as input  
**Then** it should return a binary classification (0 or 1) with confidence probability

**Given** a batch of multiple grid cells  
**When** processed through the API  
**Then** all predictions should be returned within reasonable time (< 5 seconds for 100 cells)

**Given** a well-served region with high stop density and route frequency  
**When** classified by the model  
**Then** it should predict label 1 (well-served) with confidence > 70%

---

## Functional Requirements

### FR1: Geographic Grid Generation
**Priority:** High  
The system shall divide Belo Horizonte into a regular geographic grid where each cell represents an analysis unit.

**Acceptance Criteria:**
- Grid cells use square/rectangular geometry aligned with lat/lon coordinates
- Grid cells have consistent size (e.g., 500m x 500m or configurable)
- Each cell has unique identifier and geographic bounds (lat/lon)
- Grid covers the entire city area served by BHTrans
- Grid generation is reproducible with fixed seed

### FR2: Feature Extraction from GTFS Data
**Priority:** High  
The system shall extract transit coverage metrics for each grid cell from GTFS data.

**Acceptance Criteria:**
- Count of bus stops within each cell
- Count of distinct routes serving each cell
- Estimated daily trip frequency per cell
- Optional: Coverage metrics (stops per km², routes per area)
- All metrics are numeric and normalized using StandardScaler (z-score normalization)
- Missing values are handled appropriately (imputation or exclusion documented)

### FR3: Label Generation
**Priority:** High  
The system shall automatically generate binary labels based on coverage metrics.

**Acceptance Criteria:**
- Labels are assigned using a composite coverage score combining stop_count, route_count, and daily_trips metrics
- Composite score calculated as weighted average (configurable weights, default: stops 40%, routes 30%, trips 30%)
- Threshold uses quantile-based approach (e.g., top 30% = well-served, configurable)
- Threshold methodology is documented and configurable
- Label distribution is balanced enough for training (minimum 20% in minority class)
- Labels can be validated against manual inspection

### FR4: Model Training
**Priority:** High  
The system shall train multiple classification models and select the best performer.

**Acceptance Criteria:**
- At least 3 algorithms tested (Random Forest, Logistic Regression, Gradient Boosting)
- Training uses random stratified split maintaining class balance (70/15/15 for train/validation/test)
- Hyperparameter tuning performed using grid search or random search
- Model selection based on F1-score on validation set
- Cross-validation results are documented with mean and std deviation
- Training process is reproducible with random seed
- Model comparison table generated with all metrics across algorithms

### FR5: Model Evaluation
**Priority:** High  
The system shall evaluate model performance using standard metrics.

**Acceptance Criteria:**
- Reports accuracy, precision, recall, and F1-score for all tested algorithms
- Generates confusion matrix for each model
- Provides classification report for both classes
- Identifies and documents any class imbalance issues
- Validation metrics meet minimum thresholds (F1 > 0.70)
- Produces ROC curves and AUC scores for comparison
- Generates feature importance plots for interpretability
- Documents model comparison with metrics table (algorithm, accuracy, precision, recall, F1, AUC)

### FR6: Model Export
**Priority:** High  
The system shall export the best model to ONNX format.

**Acceptance Criteria:**
- Model exported to `.onnx` format
- Export includes metadata (feature names, model type, version)
- Exported model produces identical predictions to original
- File size is reasonable (< 100MB)

### FR7: API Inference Endpoint
**Priority:** Medium  
The system shall provide a REST API endpoint for model predictions.

**Acceptance Criteria:**
- Accepts JSON input with pre-computed feature vector (stop_count, route_count, daily_trips, etc.)
- Returns classification label (0 or 1) and probability score
- Handles single and batch predictions
- Returns appropriate error messages for invalid input (missing features, incorrect types)
- Response time < 200ms for single prediction
- API documentation available via /docs endpoint
- Input validation ensures feature vector matches training format

### FR8: Visualization Support
**Priority:** Low  
The system shall support visualization of classification results on a map.

**Acceptance Criteria:**
- Generates data export compatible with mapping libraries
- Includes geographic coordinates for each prediction
- Supports color-coding by classification
- Optional: Export to GeoJSON format

---

## Success Criteria

The feature will be considered successful when:

1. **Model Performance**: F1-score achieves at least 0.70 on test set for both classes
2. **Coverage Analysis**: At least 90% of the city area is successfully classified
3. **API Responsiveness**: Single predictions return in under 200ms
4. **Batch Processing**: 100 predictions complete in under 5 seconds
5. **Data Quality**: Less than 5% of grid cells have missing feature values
6. **Reproducibility**: Training pipeline can be re-run with same results using fixed seed
7. **Documentation**: All notebooks run successfully and produce expected outputs
8. **Deployment**: Model is exported to ONNX and loads successfully in API
9. **Model Comparison**: At least 3 algorithms compared with documented metrics
10. **Technical Report**: Complete PDF report documents methodology, results, and critical analysis

---

## Deliverables

As per course requirements, this feature must produce two mandatory deliverables:

### 1. GitHub Repository (Current Repository)
- ✅ README.md with problem description, folder organization, execution instructions
- ✅ Complete source code (training, export, API, preprocessing)
- ✅ Jupyter notebooks for development and analysis
- ✅ Exported models in ONNX format
- ✅ requirements.txt with all dependencies

### 2. Technical Report (PDF)
**Required Sections:**
- **Modeling Description**: Pipeline stages, algorithm choices, preprocessing steps
- **Results**: Metrics tables, confusion matrices, feature importance plots, ROC curves
- **Critical Evaluation**: Model limitations, overfitting analysis, failure cases, improvement suggestions
- **Reproduction Instructions**: Steps to reproduce experiments with exact commands and expected outputs

**Location**: `reports/transit_coverage_classifier_report.pdf`

---

## Data Requirements

### Data Sources

**Primary Source:**
- **GTFSBHTRANS.zip**: Located at `data/raw/GTFSBHTRANS.zip`
- Contains: stops.txt, routes.txt, trips.txt, stop_times.txt, shapes.txt, calendar.txt

**Processed Data:**
- Parquet files: `data/processed/gtfs/*.parquet`
- Feature dataset: `data/processed/features/grid_features.parquet`
- Labels: `data/processed/labels/grid_labels.parquet`

### Data Quality Requirements

- Stop coordinates must be within Belo Horizonte bounds
- Route data must have valid trip counts
- No duplicate stop IDs within same location
- Temporal data (schedules) should be consistent with calendar definitions

---

## Key Entities

### GridCell
Represents a geographic region for analysis.

**Attributes:**
- `cell_id` (string): Unique identifier
- `lat_min`, `lat_max` (float): Latitude bounds
- `lon_min`, `lon_max` (float): Longitude bounds  
- `centroid_lat`, `centroid_lon` (float): Cell center coordinates
- `area_km2` (float): Cell area

### TransitFeatures
Contains calculated metrics for a grid cell.

**Attributes:**
- `cell_id` (string): Reference to GridCell
- `stop_count` (int): Number of stops in cell
- `route_count` (int): Number of distinct routes
- `daily_trips` (int): Estimated trips per day
- `stop_density` (float): Stops per km²
- `route_diversity` (float): Measure of route variety

### Classification
Model prediction for a grid cell.

**Attributes:**
- `cell_id` (string): Reference to GridCell
- `label` (int): 0 (underserved) or 1 (well-served)
- `probability` (float): Confidence score [0, 1]
- `model_version` (string): Model identifier

---

## Assumptions

1. **Geographic Scope**: Analysis covers only areas within BHTrans service area
2. **Grid Size**: 500m x 500m cells provide adequate granularity (configurable)
3. **Label Threshold**: Top 30% coverage = well-served (can be adjusted based on data distribution)
4. **Data Completeness**: GTFS data is complete and representative of actual service
5. **Static Analysis**: Model analyzes current state, not temporal changes
6. **Coordinate System**: WGS84 (EPSG:4326) used for all geographic calculations
7. **Python Environment**: Python 3.10+ with scikit-learn, pandas, geopandas available

---

## Dependencies

### Technical Dependencies
- Python 3.10+
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- geopandas >= 0.14.0
- shapely >= 2.0.0
- onnx >= 1.14.0
- onnxruntime >= 1.15.0

### Data Dependencies
- GTFSBHTRANS.zip must be available and up-to-date
- Parquet conversion completed successfully
- Valid geographic bounds for Belo Horizonte

### Service Dependencies
- FastAPI server running for API inference
- Jupyter notebook environment for development

---

## Constraints

1. **Performance**: Model training should complete within 30 minutes on standard hardware
2. **Memory**: Peak memory usage should not exceed 8GB during training
3. **Model Size**: Exported ONNX model should be under 100MB
4. **API Latency**: Individual predictions must return within 200ms
5. **Data Volume**: System must handle at least 10,000 grid cells
6. **Interpretability**: Model should provide feature importance scores
7. **Course Compliance**: Must fulfill all requirements of UECE ML course final project
8. **Reproducibility**: All experiments must be reproducible with documented random seeds and environment specifications

---

## Edge Cases

1. **Empty Grid Cells**: Cells with no transit coverage (0 stops, 0 routes)
   - **Handling**: Assign default low values, label as underserved
   
2. **Boundary Cells**: Cells partially outside service area
   - **Handling**: Include if > 50% within bounds, otherwise exclude

3. **Overlapping Routes**: Multiple routes using same stops
   - **Handling**: Count unique routes, not route instances

4. **Temporal Variation**: Service frequency varies by day/time
   - **Handling**: Use average weekday frequency as baseline

5. **Incomplete GTFS Data**: Missing shape or schedule information
   - **Handling**: Log warnings, use available data, mark uncertainty

6. **Extreme Values**: Cells with unusually high stop density (e.g., terminals)
   - **Handling**: Apply outlier detection, cap at 95th percentile if needed

---

## Non-Functional Requirements

### Performance
- Feature extraction: Process 10,000 cells in under 5 minutes
- Model training: Complete in under 30 minutes
- Prediction latency: < 200ms per request
- Batch predictions: 100 cells in < 5 seconds

### Scalability
- Support up to 20,000 grid cells
- Handle concurrent API requests (10+ simultaneous)
- Model retraining with updated data quarterly

### Maintainability
- Code follows PEP 8 standards
- Functions are documented with docstrings
- Notebooks have clear markdown explanations
- Configuration parameters externalized

### Reliability
- Model predictions are deterministic with fixed seed
- API handles errors gracefully with appropriate status codes
- Data validation catches common errors
- Training pipeline includes checkpoints

---

## Clarifications

### Session 2025-12-09

- Q: Grid generation method - square/rectangular, hexagonal (H3), or adaptive sizing? → A: Square/rectangular grids aligned with lat/lon coordinates
- Q: Feature normalization strategy - StandardScaler, MinMaxScaler, RobustScaler, or none? → A: StandardScaler (z-score normalization, mean=0, std=1)
- Q: Label threshold methodology - single metric, composite score, multi-criteria, or hierarchical? → A: Composite score - weighted combination of stop_count, route_count, daily_trips
- Q: Train/validation/test split strategy - spatial split or random stratified split? → A: Random stratified split maintaining class balance
- Q: API input format - accept coordinates, pre-computed features, or hybrid? → A: Accept pre-computed features (feature vector directly)

## Open Questions

None at this time. All aspects of the specification are clear and actionable.

---

## Future Enhancements

These are explicitly out of scope for the initial implementation but may be considered later:

1. **Demographic Integration**: Incorporate population density and socioeconomic data
2. **Multi-class Classification**: Classify into 3+ service levels (poor/fair/good/excellent)
3. **Temporal Analysis**: Track coverage changes over time
4. **Multi-city Support**: Extend to other Brazilian cities with GTFS data
5. **Real-time Updates**: Integrate with real-time transit APIs
6. **Interactive Dashboard**: Web-based visualization tool for stakeholders
7. **Alternative Grid Systems**: Support for hexagonal (H3) or administrative boundaries
8. **Accessibility Features**: Include wheelchair access and other accessibility metrics

---

## References

- GTFS Specification: https://gtfs.org/
- BHTrans Open Data: (if publicly available)
- Similar Studies: (academic papers on transit equity analysis)
- H3 Hexagonal Grid System: https://h3geo.org/

---

## Approval

**Stakeholders:**
- [ ] Project Owner
- [ ] Technical Lead
- [ ] Data Scientist
- [ ] Urban Planning Representative (if applicable)

**Status:** Ready for Implementation

---

*This specification focuses on "what" needs to be built and "why" it matters, without prescribing specific implementation details (the "how"). Technical decisions should be made during development based on this specification.*
