# Tasks: Transit Coverage Classifier

**Input**: Design documents from `/specs/1-transit-coverage-classifier/`  
**Prerequisites**: plan.md, spec.md, research.md  
**Branch**: `1-transit-coverage-classifier`

**Tests**: Tests are OPTIONAL for this academic ML project and are NOT included unless explicitly requested.

**Organization**: Tasks are organized by ML pipeline stage (treated as user stories) to enable independent implementation and testing.

## Format: `- [ ] [ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which pipeline stage/user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency installation

- [X] T001 Install new dependencies (geopandas>=0.14.0, shapely>=2.0.0, pyproj>=3.6.0) in requirements.txt
- [X] T002 Create config/model_config.yaml with grid, feature, label, and training parameters from research.md
- [X] T003 Update .github/copilot-instructions.md with geopandas, shapely, StandardScaler context via update-agent-context.sh

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures and configuration that ALL pipeline stages depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Create data/processed/grids/ directory structure
- [X] T005 Create data/processed/features/ directory structure
- [X] T006 Create data/processed/labels/ directory structure
- [X] T007 Create models/transit_coverage/ directory structure
- [X] T008 Create tests/unit/ directory structure
- [X] T009 Create tests/integration/ directory structure
- [X] T010 Create tests/fixtures/sample_gtfs/ directory structure

**Checkpoint**: Foundation ready - pipeline implementation can now begin in parallel

---

## Phase 3: Grid Generation (Priority: P1) 🎯 MVP

**Goal**: Generate 500m×500m geographic grid covering Belo Horizonte transit service area (FR1)

**Independent Test**: Verify grid has ~3,520 cells covering bounds (-20.046° to -19.758° lat, -44.081° to -43.844° lon)

### Implementation

- [X] T011 [US1] Create GridGenerator class in src/data/grid_generator.py with generate_grid() method
- [X] T012 [US1] Implement lat/lon bounds calculation using research findings (lat: -20.046 to -19.758, lon: -44.081 to -43.844) in src/data/grid_generator.py
- [X] T013 [US1] Implement 500m×500m cell generation using geopandas and shapely.geometry.box in src/data/grid_generator.py
- [X] T014 [US1] Add cell_id generation (format: "cell_{row}_{col}") and centroid calculation in src/data/grid_generator.py
- [X] T015 [US1] Save grid to data/processed/grids/cells.parquet with columns: cell_id, lat_min, lat_max, lon_min, lon_max, centroid_lat, centroid_lon, area_km2, geometry
- [X] T016 [US1] Add validation to ensure ~3,520 cells generated and all cells have area ≈ 0.25 km² in src/data/grid_generator.py

**Checkpoint**: Grid generation complete - can visualize cells and verify coverage

---

## Phase 4: Feature Extraction (Priority: P1) 🎯 MVP

**Goal**: Extract transit coverage metrics (stop_count, route_count, daily_trips) for each grid cell (FR2)

**Independent Test**: Verify features extracted for all cells with average 2.82 stops/cell

### Implementation

- [X] T017 [US2] Create FeatureExtractor class in src/data/feature_extractor.py with extract_features() method
- [X] T018 [US2] Load GTFS stops from data/processed/gtfs/stops.parquet and convert to GeoDataFrame in src/data/feature_extractor.py
- [X] T019 [US2] Implement spatial join to assign stops to grid cells using geopandas.sjoin in src/data/feature_extractor.py
- [X] T020 [US2] Calculate stop_count per cell (count of stops within each cell) in src/data/feature_extractor.py
- [X] T021 [US2] Load stop_times and trips from parquet, join to get route_id per stop, calculate route_count per cell in src/data/feature_extractor.py
- [X] T022 [US2] Implement daily trip frequency calculation using research methodology (filter weekday services from calendar, count unique trips per stop, aggregate to cells) in src/data/feature_extractor.py
- [X] T023 [US2] Calculate optional metrics (stop_density = stops/km², route_diversity) in src/data/feature_extractor.py
- [X] T024 [US2] Save raw features to data/processed/features/grid_features_raw.parquet
- [X] T025 [US2] Implement StandardScaler normalization (fit on training set, transform val/test) in src/data/feature_extractor.py
- [X] T026 [US2] Save normalized features with columns: cell_id, stop_count, route_count, daily_trips, stop_count_norm, route_count_norm, daily_trips_norm to data/processed/features/grid_features.parquet
- [X] T027 [US2] Save fitted scaler to models/transit_coverage/scaler.pkl for inference

**Checkpoint**: Features extracted and normalized - ready for label generation

---

## Phase 5: Label Generation (Priority: P1) 🎯 MVP

**Goal**: Generate binary labels (0=underserved, 1=well-served) using composite score and 70th percentile threshold (FR3)

**Independent Test**: Verify ~30% cells labeled as well-served (1,056 cells), ~70% underserved (2,464 cells)

### Implementation

- [X] T028 [US3] Create LabelGenerator class in src/data/feature_extractor.py or separate module with generate_labels() method
- [X] T029 [US3] Load normalized features from data/processed/features/grid_features.parquet
- [X] T030 [US3] Calculate composite score using weights from config (40% stops, 30% routes, 30% trips): score = 0.4*stop_count_norm + 0.3*route_count_norm + 0.3*daily_trips_norm
- [X] T031 [US3] Calculate 70th percentile threshold from composite scores
- [X] T032 [US3] Assign binary labels: 1 if score >= threshold, 0 otherwise
- [X] T033 [US3] Validate label distribution (minority class >= 20%, target ~30%)
- [X] T034 [US3] Save labels to data/processed/labels/grid_labels.parquet with columns: cell_id, composite_score, label, threshold_used, weights

**Checkpoint**: Labels generated and validated - ready for model training

---

## Phase 6: Dataset Preparation (Priority: P1) 🎯 MVP

**Goal**: Merge features and labels, perform stratified split (70/15/15) maintaining class balance

**Independent Test**: Verify split produces train=2,464, val=528, test=528 cells with balanced classes

### Implementation

- [X] T035 [US4] Create DatasetPreparator class in src/data/preprocessing.py with prepare_dataset() method
- [X] T036 [US4] Load features from data/processed/features/grid_features.parquet and labels from data/processed/labels/grid_labels.parquet
- [X] T037 [US4] Merge features and labels on cell_id
- [X] T038 [US4] Implement stratified split using sklearn.model_selection.train_test_split with random_state=42, test_size=0.15, stratify=labels
- [X] T039 [US4] Split remaining into train/val with stratify, test_size=0.176 (to get 15% of original)
- [X] T040 [US4] Validate split sizes and class balance in each set
- [X] T041 [US4] Save splits to data/processed/features/train.parquet, data/processed/features/val.parquet, data/processed/features/test.parquet

**Checkpoint**: Dataset prepared and split - ready for model training

---

## Phase 7: Model Training (Priority: P1) 🎯 MVP

**Goal**: Train 3 classification models (Random Forest, Logistic Regression, Gradient Boosting) with hyperparameter tuning (FR4)

**Independent Test**: Verify all 3 models trained, cross-validation complete, best model selected based on F1-score

### Implementation

- [X] T042 [US5] Create ModelTrainer class in src/models/train.py with train_all_models() method
- [X] T043 [US5] Load train/val datasets from data/processed/features/train.parquet and val.parquet
- [X] T044 [US5] Implement Logistic Regression training with GridSearchCV (4 combinations: C=[0.01, 0.1, 1.0, 10.0], cv=5, scoring='f1') in src/models/train.py
- [X] T045 [US5] Implement Random Forest training with RandomizedSearchCV (n_iter=20, param_dist from research.md, cv=5, scoring='f1', n_jobs=-1) in src/models/train.py
- [X] T046 [US5] Implement Gradient Boosting training with RandomizedSearchCV (n_iter=15, param_dist from research.md, cv=5, scoring='f1', n_jobs=-1) in src/models/train.py
- [X] T047 [US5] Save all trained models to models/transit_coverage/ with filenames: logistic_regression.pkl, random_forest.pkl, gradient_boosting.pkl
- [X] T048 [US5] Log hyperparameter search results (best params, CV scores) for each model
- [X] T049 [US5] Select best model based on validation F1-score and save as models/transit_coverage/best_model.pkl
- [X] T050 [US5] Validate training time constraint (~40 minutes total, optimize if exceeding 30 minutes by reducing CV folds to 3)

**Checkpoint**: ✅ All models trained and best model selected - ready for evaluation

---

## Phase 8: Model Evaluation (Priority: P1) 🎯 MVP

**Goal**: Evaluate model performance using standard metrics, generate confusion matrices, ROC curves, feature importance plots (FR5)

**Independent Test**: Verify F1-score >= 0.70 on test set, all metrics documented

### Implementation

- [X] T051 [P] [US6] Create ModelEvaluator class in src/models/evaluator.py with evaluate_all_models() method
- [X] T052 [P] [US6] Load test dataset from data/processed/features/test.parquet in src/models/evaluator.py
- [X] T053 [US6] Implement evaluation metrics calculation (accuracy, precision, recall, F1-score, ROC-AUC) for all 3 models in src/models/evaluator.py
- [X] T054 [US6] Generate confusion matrix for each model and save to reports/figures/confusion_matrix_{model_name}.png
- [X] T055 [US6] Generate ROC curves for all models on same plot and save to reports/figures/roc_curves_comparison.png
- [X] T056 [US6] Extract feature importance from each model (RF/GB: feature_importances_, LR: coefficient magnitude) in src/models/evaluator.py
- [X] T057 [US6] Generate feature importance comparison plot (side-by-side bar charts) and save to reports/figures/feature_importance_comparison.png
- [X] T058 [US6] Create metrics comparison table with columns: algorithm, accuracy, precision, recall, f1, auc and save to reports/tables/model_comparison.csv
- [X] T059 [US6] Generate classification report for best model (per-class precision, recall, F1) and save to reports/tables/classification_report.txt
- [X] T060 [US6] Validate F1-score >= 0.70 threshold for best model on test set

**Checkpoint**: ✅ All evaluation complete, metrics documented - ready for model export

---

## Phase 9: Model Export (Priority: P1) 🎯 MVP

**Goal**: Export best model to ONNX format for efficient inference (FR6)

**Independent Test**: Verify ONNX model produces identical predictions to original model, file size < 100MB

### Implementation

- [X] T061 [US7] Load best model from models/transit_coverage/best_model.pkl in src/models/export.py
- [X] T062 [US7] Convert model to ONNX format using skl2onnx with feature names and metadata in src/models/export.py
- [X] T063 [US7] Save ONNX model to models/transit_coverage/best_model.onnx
- [X] T064 [US7] Validate ONNX model predictions match original model on sample data in src/models/export.py
- [X] T065 [US7] Verify ONNX file size < 100MB constraint
- [X] T066 [US7] Save model metadata (model_type, feature_names, model_version, training_date) to models/transit_coverage/model_metadata.json

**Checkpoint**: ✅ ONNX model exported and validated - ready for API serving

---

## Phase 10: API Inference Endpoint (Priority: P2)

**Goal**: Provide REST API endpoint for model predictions (single and batch) (FR7)

**Independent Test**: Verify API responds with predictions, latency < 200ms for single prediction

### Implementation

- [X] T067 [P] [US8] Create PredictionService class in src/api/prediction_service.py with load_model() and predict() methods
- [X] T068 [P] [US8] Load ONNX model and scaler in src/api/prediction_service.py using onnxruntime.InferenceSession
- [X] T069 [US8] Implement POST /predict endpoint in src/api/main.py accepting JSON with features (stop_count_norm, route_count_norm, daily_trips_norm)
- [X] T070 [US8] Implement POST /predict/batch endpoint in src/api/main.py accepting array of predictions with cell_id and features
- [X] T071 [US8] Implement GET /health endpoint in src/api/main.py returning status, model_loaded, model_version
- [X] T072 [US8] Add input validation for missing features and incorrect types in src/api/main.py
- [X] T073 [US8] Add error handling with appropriate HTTP status codes (400, 500) in src/api/main.py
- [X] T074 [US8] Test single prediction latency (target < 200ms) using sample requests
- [X] T075 [US8] Test batch prediction performance (100 predictions in < 5 seconds)
- [X] T076 [US8] Update API documentation at /docs endpoint with request/response schemas

**Checkpoint**: ✅ API functional and performance validated - ready for technical report

---

## Phase 11: Technical Report (Priority: P1) 🎯 MVP

**Goal**: Generate PDF technical report documenting methodology, results, and critical evaluation (Deliverable requirement)

**Independent Test**: Verify report has all required sections, figures embedded, reproduction instructions complete

### Implementation

- [X] T077 [US9] Create reports/ directory structure with subdirectories: figures/, tables/
- [X] T078 [P] [US9] Write Section 1: Modeling Description (pipeline stages, preprocessing, algorithms, hyperparameters) in reports/transit_coverage_classifier_report.md
- [X] T079 [P] [US9] Write Section 2: Results (embed metrics tables from reports/tables/model_comparison.csv, confusion matrices, ROC curves, feature importance plots)
- [X] T080 [P] [US9] Write Section 3: Critical Evaluation (model limitations, overfitting analysis, class imbalance discussion, failure cases, improvement suggestions)
- [X] T081 [P] [US9] Write Section 4: Reproduction Instructions (exact commands to run pipeline, expected outputs, environment specifications)
- [X] T082 [US9] Convert markdown to PDF using pandoc or similar tool, save as reports/transit_coverage_classifier_report.pdf
- [X] T083 [US9] Validate PDF has all figures embedded and is readable

**Checkpoint**: ✅ Technical report complete - all deliverables ready

---

## Phase 12: Documentation & Finalization (Priority: P2)

**Goal**: Update README, create notebooks, ensure reproducibility

**Independent Test**: Verify README has complete instructions, notebooks run successfully

### Implementation

- [X] T084 [P] Update README.md with problem description, folder organization, execution instructions, and deliverables section
- [X] T085 [P] Create notebooks/02_grid_generation.ipynb demonstrating grid creation and visualization (covered by notebooks/02_feature_engineering.ipynb)
- [X] T086 [P] Create notebooks/03_feature_extraction.ipynb demonstrating feature calculation and normalization (covered by notebooks/02_feature_engineering.ipynb)
- [X] T087 [P] Create notebooks/04_model_training.ipynb demonstrating training pipeline and hyperparameter tuning (covered by notebooks/03_model_training.ipynb)
- [X] T088 [P] Update notebooks/01_exploratory_analysis.ipynb to include geographic bounds analysis from research.md
- [X] T089 Create quickstart.md with step-by-step developer guide (covered by comprehensive README.md with Quick Start Guide and PIPELINE_USAGE.md)
- [X] T090 Run complete pipeline end-to-end to validate reproducibility with fixed random seed
- [X] T091 Commit all generated artifacts (models, reports, figures) to repository
- [X] T092 Final validation: verify all success criteria met (F1>=0.70, API latency<200ms, documentation complete)

**Checkpoint**: ✅ All documentation and artifacts complete - feature ready for submission

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all pipeline stages
- **Grid Generation (Phase 3)**: Depends on Foundational completion
- **Feature Extraction (Phase 4)**: Depends on Grid Generation (needs grid cells)
- **Label Generation (Phase 5)**: Depends on Feature Extraction (needs normalized features)
- **Dataset Preparation (Phase 6)**: Depends on Label Generation (needs features + labels)
- **Model Training (Phase 7)**: Depends on Dataset Preparation (needs train/val splits)
- **Model Evaluation (Phase 8)**: Depends on Model Training (needs trained models)
- **Model Export (Phase 9)**: Depends on Model Evaluation (needs best model)
- **API Inference (Phase 10)**: Depends on Model Export (needs ONNX model)
- **Technical Report (Phase 11)**: Depends on Model Evaluation (needs results and figures)
- **Documentation (Phase 12)**: Depends on all prior phases (final integration)

### Pipeline Stage Dependencies

- **Grid Generation (US1)**: Can start after Foundational (Phase 2) - No dependencies on other stages
- **Feature Extraction (US2)**: Depends on Grid Generation (US1) completing
- **Label Generation (US3)**: Depends on Feature Extraction (US2) completing
- **Dataset Preparation (US4)**: Depends on Label Generation (US3) completing
- **Model Training (US5)**: Depends on Dataset Preparation (US4) completing
- **Model Evaluation (US6)**: Depends on Model Training (US5) completing
- **Model Export (US7)**: Depends on Model Evaluation (US6) completing
- **API Inference (US8)**: Depends on Model Export (US7) completing (can proceed in parallel with Technical Report)
- **Technical Report (US9)**: Depends on Model Evaluation (US6) completing (can proceed in parallel with API)

### Parallel Opportunities

- **Phase 1 (Setup)**: T001, T002, T003 can run in parallel (different files)
- **Phase 2 (Foundational)**: T004-T010 can run in parallel (directory creation)
- **Phase 8 (Evaluation)**: T051, T052 can run in parallel (different concerns)
- **Phase 8 (Evaluation)**: T054, T055, T056, T057 can run in parallel (independent visualizations)
- **Phase 10 (API)**: T067, T068 can run in parallel (different modules)
- **Phase 11 (Report)**: T078, T079, T080, T081 can run in parallel (different sections)
- **Phase 12 (Documentation)**: T084, T085, T086, T087, T088 can run in parallel (independent notebooks)
- **Cross-phase**: Phase 10 (API) and Phase 11 (Report) can proceed in parallel after Phase 9 completes

---

## Parallel Example: Feature Extraction (Phase 4)

```bash
# These tasks must run sequentially as they depend on each other:
Task T017: Create FeatureExtractor class
Task T018: Load GTFS stops and convert to GeoDataFrame
Task T019: Implement spatial join
Task T020: Calculate stop_count
Task T021: Calculate route_count
Task T022: Calculate daily_trips
Task T023: Calculate optional metrics
Task T024: Save raw features
Task T025: Implement StandardScaler
Task T026: Save normalized features
Task T027: Save scaler

# But within Phase 8 (Evaluation), these can run in parallel:
Task T054: Generate confusion matrices (all models)
Task T055: Generate ROC curves
Task T056: Extract feature importance
Task T057: Generate feature importance plots
```

---

## Implementation Strategy

### MVP First (Core Pipeline)

1. Complete Phase 1: Setup (dependencies, config)
2. Complete Phase 2: Foundational (directory structure) - CRITICAL
3. Complete Phase 3: Grid Generation (US1)
4. Complete Phase 4: Feature Extraction (US2)
5. Complete Phase 5: Label Generation (US3)
6. Complete Phase 6: Dataset Preparation (US4)
7. Complete Phase 7: Model Training (US5)
8. Complete Phase 8: Model Evaluation (US6)
9. Complete Phase 9: Model Export (US7)
10. Complete Phase 11: Technical Report (US9) - REQUIRED DELIVERABLE
11. **STOP and VALIDATE**: Verify F1>=0.70, all metrics documented, report complete

### Incremental Delivery

1. Complete Setup + Foundational → Infrastructure ready
2. Add Grid Generation → Visualize grid cells
3. Add Feature Extraction → Validate features (2.82 stops/cell avg)
4. Add Label Generation → Validate distribution (30% well-served)
5. Add Dataset Preparation → Verify splits (2464/528/528)
6. Add Model Training → Train all 3 models
7. Add Model Evaluation → Compare models, select best
8. Add Model Export → ONNX ready
9. Add Technical Report → Complete required deliverable
10. Add API (optional) → Enable predictions
11. Add Documentation → Finalize submission

### Sequential Execution (Single Developer)

This is an ML pipeline with strong sequential dependencies:

1. Foundation setup (Phases 1-2)
2. Data pipeline (Phases 3-6): Grid → Features → Labels → Splits
3. ML pipeline (Phases 7-9): Train → Evaluate → Export
4. Deliverables (Phases 10-12): API, Report, Documentation
5. Each phase must complete before next can begin
6. Use parallel tasks within phases where marked [P]

---

## Notes

- [P] tasks = different files, no dependencies within the phase
- [Story] label maps task to specific pipeline stage (US1-US9)
- Each pipeline stage builds on previous stage - strict sequential dependencies
- Research findings guide implementation (bounds, grid size, thresholds from research.md)
- All numeric values (bounds, thresholds, weights) come from research.md
- Training time target: ~40 minutes (may need CV fold reduction to meet 30-min constraint)
- Expected performance: F1 0.72-0.83, API latency 15-20ms
- Tests NOT included as per specification (not requested for academic ML project)
- Focus on reproducibility: fixed seeds, documented parameters, clear instructions

---

## Summary

**Total Tasks**: 92  
**Pipeline Stages (User Stories)**: 9  
- US1: Grid Generation (6 tasks)
- US2: Feature Extraction (11 tasks)
- US3: Label Generation (7 tasks)
- US4: Dataset Preparation (7 tasks)
- US5: Model Training (9 tasks)
- US6: Model Evaluation (10 tasks)
- US7: Model Export (6 tasks)
- US8: API Inference (10 tasks)
- US9: Technical Report (7 tasks)

**Parallel Opportunities**: 23 tasks marked [P] can run in parallel within their phases

**Suggested MVP Scope**: Phases 1-9 + Phase 11 (Technical Report) = 76 tasks  
**Optional Extensions**: Phase 10 (API) + Phase 12 (Documentation) = 16 additional tasks

**Critical Path**: Setup → Foundational → Grid → Features → Labels → Dataset → Training → Evaluation → Export → Report
