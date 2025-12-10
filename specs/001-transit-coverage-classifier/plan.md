# Implementation Plan: Transit Coverage Classifier

**Branch**: `1-transit-coverage-classifier` | **Date**: 2025-12-09 | **Spec**: [spec.md](./spec.md)

## Summary

Develop a binary classification ML model to identify well-served vs underserved regions in Belo Horizonte based on public transit coverage metrics extracted from GTFS data. The system will generate a geographic grid, extract transit features (stop count, route count, trip frequency), train multiple classification models (Random Forest, Logistic Regression, Gradient Boosting), export the best model to ONNX format, and serve predictions via a FastAPI endpoint.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: scikit-learn 1.3.0+, pandas 2.0.0+, geopandas 0.14.0+, FastAPI 0.104.0+, ONNX Runtime 1.15.0+  
**Storage**: Parquet files (data/processed/gtfs/, data/processed/features/, data/processed/labels/)  
**Testing**: pytest for unit and integration tests  
**Target Platform**: Linux server (development), compatible with standard ML environments  
**Project Type**: Single project - ML pipeline with API serving  
**Performance Goals**: 
- Feature extraction: 10,000 cells in < 5 minutes
- Model training: complete in < 30 minutes
- API latency: < 200ms per prediction
- Batch processing: 100 predictions in < 5 seconds

**Constraints**:
- Memory: < 8GB during training
- Model size: < 100MB (ONNX export)
- F1-score: ≥ 0.70 on test set
- Data quality: < 5% missing values in grid cells

**Scale/Scope**: 
- Dataset: ~10,000 grid cells covering Belo Horizonte
- GTFS data: 9,917 stops, 323 routes, 51,122 trips, 2.9M stop_times records
- Output: PDF technical report + GitHub repository

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✅ PASSED

This is an academic ML project with no organizational constitution defined. The project follows standard ML best practices:
- Self-contained Python project with clear module separation
- Reproducible pipeline with documented random seeds
- Standard test framework (pytest)
- API follows REST conventions with OpenAPI documentation
- All code follows PEP 8 standards

No constitution violations identified.

## Project Structure

### Documentation (this feature)

```text
specs/1-transit-coverage-classifier/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file - implementation plan
├── research.md          # Phase 0 output - technical research
├── data-model.md        # Phase 1 output - entity definitions
├── quickstart.md        # Phase 1 output - developer guide
├── contracts/           # Phase 1 output - API schemas
└── checklists/
    └── requirements.md  # Quality validation (completed)
```

### Source Code (repository root)

```text
# ML Project Structure (existing + enhancements)
src/
├── data/
│   ├── gtfs_loader.py          # ✅ Exists - loads GTFS data
│   ├── preprocessing.py        # ✅ Exists - data preprocessing
│   ├── grid_generator.py       # ⚠️ NEW - generate geographic grids
│   └── feature_extractor.py    # ⚠️ NEW - extract transit features
├── features/
│   └── feature_engineering.py  # ✅ Exists - feature engineering
├── models/
│   ├── train.py                # ✅ Exists - model training
│   ├── export.py               # ✅ Exists - ONNX export
│   └── evaluator.py            # ⚠️ NEW - model evaluation & comparison
└── api/
    └── main.py                 # ✅ Exists - FastAPI application

data/
├── raw/
│   └── GTFSBHTRANS.zip         # ✅ Exists - GTFS source data
├── processed/
│   ├── gtfs/                   # ✅ Exists - Parquet files
│   ├── grids/                  # ⚠️ NEW - grid definitions
│   ├── features/               # ⚠️ NEW - extracted features
│   └── labels/                 # ⚠️ NEW - classification labels

models/
└── transit_coverage/           # ⚠️ NEW - trained models
    ├── best_model.onnx
    ├── random_forest.pkl
    ├── logistic_regression.pkl
    ├── gradient_boosting.pkl
    └── scaler.pkl

notebooks/
├── 01_exploratory_analysis.ipynb     # ✅ Exists - EDA
├── 02_feature_engineering.ipynb      # ✅ Exists - feature engineering
└── 03_model_training.ipynb           # ✅ Exists - model training

reports/
└── transit_coverage_classifier_report.pdf  # ⚠️ NEW - technical report

tests/
├── unit/                       # ⚠️ NEW - unit tests
│   ├── test_grid_generator.py
│   ├── test_feature_extractor.py
│   └── test_evaluator.py
├── integration/                # ⚠️ NEW - integration tests
│   ├── test_pipeline.py
│   └── test_api.py
└── fixtures/                   # ⚠️ NEW - test data
    └── sample_gtfs/
```

**Structure Decision**: Single project structure is appropriate for this ML pipeline. The existing structure is well-organized with clear separation between data processing, feature engineering, modeling, and API serving. New components integrate naturally into existing modules.

## Complexity Tracking

No constitution violations - no complexity justification required.

---

## Phase 0: Outline & Research

**Goal**: Resolve all technical unknowns and establish implementation approach.

### Research Questions

1. **Geographic Grid Generation**
   - ✅ RESOLVED: Square/rectangular grids aligned with lat/lon coordinates
   - Implementation: Use geopandas to create grid polygons
   - Libraries: shapely.geometry.box, geopandas.GeoDataFrame

2. **Belo Horizonte Geographic Bounds**
   - NEEDS RESEARCH: Determine min/max lat/lon for BH service area
   - Source: Extract from stops.parquet or use known BH boundaries
   - Buffer: Add margin to ensure complete coverage

3. **Optimal Grid Cell Size**
   - NEEDS RESEARCH: Validate 500m x 500m assumption
   - Approach: Test multiple sizes (250m, 500m, 1km) for coverage vs granularity
   - Criteria: Balance between detail and computational cost

4. **Daily Trip Frequency Calculation**
   - NEEDS RESEARCH: Define "daily trip" from GTFS stop_times
   - Approach: Count unique trips per weekday, average across week
   - Data: Use calendar.txt to identify weekday service

5. **Feature Normalization Implementation**
   - ✅ RESOLVED: StandardScaler from scikit-learn
   - Implementation: Fit on training set, transform val/test
   - Save scaler for inference

6. **Label Distribution Analysis**
   - NEEDS RESEARCH: Validate 30% threshold produces balanced classes
   - Approach: Test thresholds (20%, 30%, 40%) for class balance
   - Criteria: Minimum 20% in minority class

7. **Hyperparameter Search Strategy**
   - NEEDS RESEARCH: GridSearchCV vs RandomizedSearchCV
   - Decision factors: Parameter space size, computational budget
   - Recommended: Start with RandomizedSearchCV for efficiency

8. **Model Interpretability Methods**
   - NEEDS RESEARCH: Feature importance extraction for each algorithm
   - Random Forest: feature_importances_ attribute
   - Logistic Regression: coefficients
   - Gradient Boosting: feature_importances_ attribute

### Research Outputs

**Deliverable**: `research.md` document containing:
- Geographic bounds for Belo Horizonte (lat/lon min/max with source)
- Grid size recommendation with justification
- Daily trip frequency calculation methodology
- Label threshold analysis with distribution plots
- Hyperparameter search space definitions for each algorithm
- Feature importance extraction code examples
- Performance baseline estimates

---

## Phase 1: Design & Contracts

**Prerequisites**: Phase 0 research completed

**Goal**: Define data models, API contracts, and architectural decisions without implementation.

### 1.1 Data Model Design

**Deliverable**: `data-model.md`

**Entities**:

1. **GridCell**
   ```python
   {
       "cell_id": str,              # Format: "cell_{row}_{col}"
       "lat_min": float,
       "lat_max": float,
       "lon_min": float,
       "lon_max": float,
       "centroid_lat": float,
       "centroid_lon": float,
       "area_km2": float,
       "geometry": Polygon          # shapely geometry
   }
   ```

2. **TransitFeatures**
   ```python
   {
       "cell_id": str,
       "stop_count": int,           # Raw count
       "route_count": int,          # Unique routes
       "daily_trips": float,        # Average daily trips
       "stop_density": float,       # Stops per km²
       "route_diversity": float,    # Shannon entropy or count
       "stop_count_norm": float,    # Normalized features
       "route_count_norm": float,
       "daily_trips_norm": float
   }
   ```

3. **ClassificationLabel**
   ```python
   {
       "cell_id": str,
       "composite_score": float,    # Weighted combination
       "label": int,                # 0=underserved, 1=well-served
       "threshold_used": float,     # Quantile threshold
       "weights": dict              # {"stops": 0.4, "routes": 0.3, "trips": 0.3}
   }
   ```

4. **ModelPrediction**
   ```python
   {
       "cell_id": str,
       "label": int,
       "probability": float,        # Confidence [0, 1]
       "model_version": str,        # e.g., "random_forest_v1"
       "timestamp": datetime
   }
   ```

### 1.2 API Contract Design

**Deliverable**: `contracts/prediction_api.yaml` (OpenAPI 3.0 schema)

**Endpoints**:

1. **POST /predict** - Single prediction
   ```yaml
   Request:
     {
       "features": {
         "stop_count_norm": float,
         "route_count_norm": float,
         "daily_trips_norm": float
       }
     }
   Response:
     {
       "label": int,
       "probability": float,
       "model_version": str
     }
   ```

2. **POST /predict/batch** - Batch predictions
   ```yaml
   Request:
     {
       "predictions": [
         {"cell_id": str, "features": {...}},
         ...
       ]
     }
   Response:
     {
       "predictions": [
         {"cell_id": str, "label": int, "probability": float},
         ...
       ],
       "model_version": str
     }
   ```

3. **GET /health** - Health check
   ```yaml
   Response:
     {
       "status": "healthy",
       "model_loaded": bool,
       "model_version": str
     }
   ```

### 1.3 Pipeline Architecture

**Stages**:
1. Grid Generation → GridCell entities saved to `data/processed/grids/cells.parquet`
2. Feature Extraction → TransitFeatures saved to `data/processed/features/grid_features.parquet`
3. Label Generation → ClassificationLabel saved to `data/processed/labels/grid_labels.parquet`
4. Dataset Preparation → Merge features + labels, stratified split
5. Model Training → Train 3 algorithms, hyperparameter tuning, cross-validation
6. Model Evaluation → Metrics, confusion matrices, ROC curves, feature importance
7. Model Export → Save best model as ONNX, save scaler
8. API Serving → Load ONNX model, FastAPI endpoints

### 1.4 Configuration Management

**Deliverable**: `config/model_config.yaml`

```yaml
grid:
  cell_size_meters: 500
  buffer_km: 2.0

features:
  stop_count_weight: 0.4
  route_count_weight: 0.3
  daily_trips_weight: 0.3

labels:
  threshold_quantile: 0.70  # Top 30% = well-served
  min_minority_class_pct: 0.20

training:
  random_seed: 42
  test_size: 0.15
  val_size: 0.15
  cv_folds: 5

models:
  random_forest:
    n_estimators: [100, 200, 500]
    max_depth: [10, 20, None]
    min_samples_split: [2, 5, 10]
  
  logistic_regression:
    C: [0.01, 0.1, 1.0, 10.0]
    penalty: ['l2']
    max_iter: [1000]
  
  gradient_boosting:
    n_estimators: [100, 200]
    learning_rate: [0.01, 0.1]
    max_depth: [3, 5, 7]

evaluation:
  metrics: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
  min_f1_threshold: 0.70
```

### 1.5 Quickstart Guide

**Deliverable**: `quickstart.md`

Developer guide with:
- Prerequisites (Python 3.10+, dependencies)
- Setup instructions (virtualenv, install packages)
- Data preparation steps
- Running the pipeline (step-by-step commands)
- Training models
- Evaluating results
- Starting the API
- Running tests
- Common troubleshooting

### 1.6 Agent Context Update

**Action**: Run `.specify/scripts/bash/update-agent-context.sh copilot`

This will update `.github/copilot-instructions.md` with:
- geopandas for geographic operations
- shapely for geometric calculations
- StandardScaler for feature normalization
- stratified splitting for balanced classes
- ONNX export for model serving

---

## Phase 2: Constitution Re-validation

**Gate**: Verify design complies with all constitution principles

**Status**: ✅ PASSED (No constitution defined for this academic project)

**Design Review**:
- ✅ Clear module separation (data, features, models, api)
- ✅ Reproducible pipeline (config-driven, seeded randomness)
- ✅ API follows REST conventions
- ✅ Standard testing framework
- ✅ Documentation includes quickstart guide
- ✅ All functional requirements addressed in design

**Proceed**: Ready for Phase 3 (Task Breakdown) via `/speckit.tasks`

---

## Notes

1. **Existing Assets**: Project already has strong foundation with GTFS loader, data preprocessing, and basic ML structure. New work focuses on geographic grid generation, feature extraction, and model evaluation.

2. **Key Dependencies**: geopandas and shapely are new critical dependencies for grid generation. Ensure these are added to requirements.txt.

3. **Data Pipeline**: All intermediate outputs (grids, features, labels) saved as Parquet for reproducibility and inspection.

4. **Model Comparison**: Systematic comparison of 3 algorithms with documented metrics ensures academic rigor required for course.

5. **Testing Strategy**: Unit tests for each new module, integration tests for pipeline and API ensure quality.

6. **Report Generation**: Technical report (PDF) is separate deliverable - generated after all experiments complete.

---

**Next Step**: Execute `/speckit.tasks` to break down Phase 1 and Phase 2 into actionable implementation tasks.
