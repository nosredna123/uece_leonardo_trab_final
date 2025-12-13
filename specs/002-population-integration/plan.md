# Implementation Plan: IBGE Population Data Integration

**Branch**: `002-population-integration` | **Date**: 2025-12-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-population-integration/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Integrate IBGE Censo 2022 population data to enrich the transit coverage classification model. The current model relies solely on transit metrics (stops, routes, trips) but lacks population context, leading to potential misclassification where high-population areas with moderate transit appear well-served or low-population areas with minimal transit appear underserved. This feature adds population density as an independent feature by: (1) regenerating the analysis grid at 200m resolution to match IBGE data, (2) loading IBGE grade estatística 2022 data directly from ZIP file, (3) merging population counts with transit features by cell identifier or spatial join, (4) retraining models with the enriched dataset. Expected outcome: improved model interpretability and better-informed urban planning decisions that account for transit demand (population) versus supply (infrastructure).

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: GeoPandas 0.14+, Pandas 2.0+, Scikit-learn 1.3+, PyArrow 14.0+ (Parquet I/O), Shapely 2.0+ (spatial ops)  
**Storage**: Parquet files (data/processed/), YAML config (config/model_config.yaml), ONNX model export  
**Testing**: pytest (existing), integration tests needed for population merge validation  
**Target Platform**: Linux development environment (bash scripts), Python scientific stack  
**Project Type**: Single project - data science/ML pipeline with CLI and API components  
**Performance Goals**: 
  - Grid regeneration + feature extraction + population merge < 10 minutes for ~15,000-20,000 200m cells
  - Population aggregation alone < 5 minutes (FR-017)
  - Model training time maintained or reduced despite increased cell count  
**Constraints**: 
  - Memory: Must handle 200m grid (~15k-20k cells) with GeoPandas in-memory operations
  - File I/O: Direct ZIP reading required (no extraction) for IBGE data
  - Data integrity: Total population sum must match Belo Horizonte census ±10%
  - CRS compatibility: Must detect and reproject CRS mismatches automatically  
**Scale/Scope**: 
  - ~15,000-20,000 grid cells at 200m resolution (increased from ~7,000 at 250m)
  - Population range: 0-3,000 per 200m cell (smaller cells = lower max population)
  - Total population: ~2.3-2.5 million (Belo Horizonte metro area)
  - Approximately 5-8 Python modules affected (grid generator, feature extractor, population integrator, model trainer, report generator)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Note**: The project constitution file (`.specify/memory/constitution.md`) is currently a template with placeholder principles. In the absence of project-specific governance rules, the following standard best practices are applied:

### Standard Quality Gates

1. **Modularity**: New functionality (population integration) MUST be implemented as reusable functions in a dedicated module (`src/data/population_integrator.py`) rather than inline scripts.
   - ✅ PASS: Spec requires FR-013 - dedicated Python module

2. **Testing**: Integration points (population merge with transit features) MUST have validation tests to verify data integrity.
   - ✅ PASS: Spec includes edge case testing for merge failures, zero-population cells, coordinate system mismatches

3. **Documentation**: Changes to data pipeline MUST be reflected in user-facing documentation (README, technical report).
   - ✅ PASS: FR-015 and FR-016 require updates to README and technical report generation

4. **Backward Compatibility**: Configuration changes (grid resolution) SHOULD preserve ability to regenerate old results if needed.
   - ⚠️ ADVISORY: 250m grid results will be replaced by 200m grid. Consider keeping baseline comparison data for model performance evaluation.

5. **Performance**: Data processing MUST complete within specified time constraints to ensure usability.
   - ✅ PASS: SC-002 and FR-017 enforce 10-minute total and 5-minute population aggregation limits with fail-fast behavior

**Gate Result**: **PASS** - All critical gates satisfied. Advisory note on baseline data preservation for comparison purposes.

---

### Post-Design Re-Evaluation (After Phase 1)

**Re-evaluation Date**: 2025-12-10  
**Artifacts Reviewed**: research.md, data-model.md, contracts/, quickstart.md

#### Design Complexity Assessment

1. **Modularity - REAFFIRMED ✅**
   - Designed module (`population_integrator.py`) with clear public API: `load_ibge_data()`, `merge_population()`, `validate_population_data()`
   - All functions have well-defined contracts with type hints and docstrings
   - No inline scripts or tightly coupled code

2. **Testing - REAFFIRMED ✅**
   - Contract specifies 10 unit tests covering success/failure cases
   - Integration tests defined for end-to-end pipeline validation
   - Performance tests included to enforce time thresholds

3. **Documentation - REAFFIRMED ✅**
   - Quickstart guide provides step-by-step user instructions
   - Data model document defines all schemas and transformations
   - Module contracts specify API behavior and error handling
   - Technical report updates defined in spec

4. **Backward Compatibility - ADVISORY MAINTAINED ⚠️**
   - Design includes config backup strategy (`model_config_250m_backup.yaml`)
   - Quickstart recommends archiving 250m results before regeneration
   - No automated rollback mechanism (acceptable for data science projects)

5. **Performance - REAFFIRMED ✅**
   - Research identifies performance bottlenecks: grid generation (60-90s), feature extraction (5-7 min), population merge (10-20s)
   - Total estimated time: 8-10 minutes (within 10-minute threshold)
   - Fail-fast behavior designed into pipeline with clear error messages

#### New Considerations from Design Phase

6. **Data Integrity**: Multi-layer validation strategy designed
   - ✅ PASS: Pre-merge validation (file existence, schema, value ranges)
   - ✅ PASS: Post-merge validation (coverage rate, total population sum, statistical distributions)
   - ✅ PASS: Runtime monitoring with actionable error messages

7. **Extensibility**: Design supports future enhancements
   - ✅ PASS: Configuration-driven approach allows easy parameter tuning
   - ✅ PASS: Dual merge strategy (ID36 + spatial join fallback) handles data variations
   - ✅ PASS: Normalization strategy (StandardScaler) consistent with existing features

#### Final Gate Result: **PASS WITH CONFIDENCE**

All quality gates remain satisfied after detailed design. The solution demonstrates:
- Clear separation of concerns (dedicated module with reusable functions)
- Comprehensive error handling (fail-fast with remediation steps)
- Thorough validation (pre/post-merge checks, runtime monitoring)
- User-friendly documentation (quickstart, contracts, troubleshooting)
- Performance awareness (time limits enforced, optimization guidance provided)

**Recommendation**: Proceed to Phase 2 (Task Breakdown) - implementation ready.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── data/
│   ├── __init__.py
│   ├── grid_generator.py         # Regenerate at 200m resolution
│   ├── feature_extractor.py      # Re-extract transit features for 200m grid
│   ├── population_integrator.py  # NEW: Load and merge IBGE population data
│   ├── gtfs_loader.py            # Existing GTFS data loading
│   ├── label_generator.py        # Existing label generation
│   └── preprocessing.py          # Existing preprocessing utilities
├── models/
│   ├── train.py                  # Update to include population feature
│   ├── evaluator.py              # Existing model evaluation
│   └── export.py                 # Existing ONNX export
├── features/
│   └── feature_engineering.py    # May need updates for population normalization
└── api/
    ├── main.py                   # API (no changes for this feature)
    └── prediction_service.py     # Prediction service (no changes)

config/
└── model_config.yaml             # Update: grid.cell_size_meters = 200

data/
├── raw/
│   ├── GTFSBHTRANS.zip           # Existing transit data
│   └── ibge_populacao_bh_grade_id36.zip  # NEW: IBGE population data (required)
└── processed/
    ├── features/
    │   └── grid_features.parquet # Updated with population column
    ├── grids/                    # Regenerated 200m grids
    └── gtfs/                     # Existing processed GTFS

tests/
├── unit/
│   └── test_population_integrator.py  # NEW: Unit tests for population module
└── integration/
    └── test_population_pipeline.py    # NEW: End-to-end population integration test

notebooks/
├── 01_exploratory_analysis.ipynb     # May update to show population distribution
├── 02_feature_engineering.ipynb      # Update to demonstrate 200m grid + population
└── 03_model_training.ipynb           # Update to show population feature impact

scripts/
├── run_pipeline.sh                    # Update to include population integration step
└── setup.sh                           # Existing setup

reports/
├── relatorio_tecnico.md               # Update via generate_report.py
└── generate_report.py                 # Update: add population integration section
```

**Structure Decision**: Single project structure (Option 1) is appropriate. This is a data science/ML pipeline project with clear separation between data processing (`src/data/`), model training (`src/models/`), API serving (`src/api/`), and analysis (`notebooks/`). The population integration feature adds one new module (`population_integrator.py`) and updates existing pipeline components without requiring architectural changes.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**N/A** - No constitution violations identified. All standard quality gates passed.
