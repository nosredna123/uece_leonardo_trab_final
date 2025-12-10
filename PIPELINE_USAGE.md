# Pipeline Execution Guide

## 🚀 Quick Start

Run the complete ML pipeline with a single command:

```bash
./run_pipeline.sh
```

## 📝 What It Does

The script executes the **complete 8-step pipeline**:

1. **Spatial Grid Generation** - Creates grid cells based on configured resolution
2. **Feature Extraction** - Extracts transit coverage features from GTFS data
3. **Label Generation** - Generates binary classification labels
4. **Data Preprocessing** - Creates train/validation/test splits
5. **Model Training** - Trains Logistic Regression, Random Forest, and Gradient Boosting
6. **Model Evaluation** - Evaluates models and generates performance metrics
7. **Model Export** - Exports best model to ONNX format
8. **Report Generation** - Creates technical report and visualizations

## ⚙️ Configuration

Before running, you can customize the pipeline in `config/model_config.yaml`:

### Grid Resolution (Most Important)
```yaml
grid:
  cell_size_meters: 150  # Options: 100, 150, 200, 250, 500...
```

**Grid Size Impact**:
- **100m**: Very fine resolution (~81,000 cells) - computationally expensive
- **150m** ⭐: Best balance (~36,000 cells, F1≈0.75-0.85) - **RECOMMENDED**
- **200m**: Conservative (~20,000 cells, F1≈0.80-0.90)
- **250m**: Moderate (~13,000 cells, F1≈0.85-0.92)
- **500m**: Coarse resolution (3,250 cells, F1≈0.98+) - too easy

### Other Settings
```yaml
model:
  random_state: 42
  test_size: 0.15
  validation_size: 0.15
  
  logistic_regression:
    max_iter: 1000
    
  random_forest:
    n_estimators: 100
    max_depth: 10
    
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
```

## 🔄 Typical Workflow

### 1. First Run (Initial Setup)
```bash
# Use default configuration (500m grids)
./run_pipeline.sh
```

### 2. Adjust Grid Size (Common Scenario)
```bash
# Edit configuration
nano config/model_config.yaml  # Change cell_size_meters to 150

# Re-run pipeline with new grid size
./run_pipeline.sh
```

### 3. Experiment with Different Sizes
```bash
# Try 200m grids
sed -i 's/cell_size_meters: .*/cell_size_meters: 200/' config/model_config.yaml
./run_pipeline.sh

# Try 250m grids
sed -i 's/cell_size_meters: .*/cell_size_meters: 250/' config/model_config.yaml
./run_pipeline.sh
```

## ⏱️ Execution Time

Approximate execution times by grid size:

| Grid Size | Cells    | Time     | Use Case                          |
|-----------|----------|----------|-----------------------------------|
| 100m      | ~81,000  | 15-20min | Research, fine-grained analysis   |
| 150m ⭐   | ~36,000  | 5-10min  | Best balance, recommended         |
| 200m      | ~20,000  | 3-5min   | Quick iterations, prototyping     |
| 250m      | ~13,000  | 2-3min   | Very quick iterations             |
| 500m      | ~3,250   | 1-2min   | Baseline, coarse analysis         |

## 📊 Understanding Results

After execution, the script displays:

```
📊 RESULTS SUMMARY
======================================================================

150m Grid Results:
  Best Model: Random Forest
  F1-Score:   0.8234
  Accuracy:   0.8567
  Status:     ✅ Good performance - realistic for spatial classification

======================================================================
✨ Pipeline executed successfully with 150m spatial resolution!
======================================================================
```

### Performance Interpretation:
- **F1 ≥ 0.98**: ⚠️ Very high - check for data leakage or over-aggregation
- **F1 ≥ 0.90**: ✅ Excellent performance
- **F1 ≥ 0.80**: ✅ Good performance - realistic for spatial classification
- **F1 ≥ 0.70**: ✓ Acceptable performance
- **F1 < 0.70**: ⚠️ Low performance - consider larger grids or feature engineering

## 📁 Generated Artifacts

After successful execution:

```
data/processed/
├── grids/cells.parquet          # Spatial grid with geometries
├── features/                    # Extracted features by type
│   ├── stop_counts.parquet
│   ├── route_counts.parquet
│   └── daily_trips.parquet
└── labels/labels.parquet        # Binary classification labels

models/transit_coverage/
├── best_model.onnx              # Production-ready model
├── best_model.pkl               # Training checkpoint
└── scaler.pkl                   # Feature normalization

reports/
├── figures/                     # Visualizations
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix_*.png
│   └── roc_curves.png
├── tables/                      # Performance metrics
│   └── model_comparison.csv
└── relatorio_tecnico.md         # Technical report (Portuguese)
```

## 🛠️ Advanced Usage

### Backup Before Running
```bash
# Manual backup
mkdir backup_$(date +%Y%m%d)
cp -r data/processed models reports backup_$(date +%Y%m%d)/

# Run pipeline
./run_pipeline.sh  # Will also prompt for backup
```

### Run Specific Steps Only
```bash
# Only regenerate grid
python -m src.data.grid_generator

# Only train models (requires existing features)
python -m src.models.train

# Only generate report
python generate_report.py
```

### Test API with Results
```bash
# Start API server
uvicorn src.api.main:app --port 8000

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": -19.9167, "longitude": -43.9345}'
```

## 🐛 Troubleshooting

### Script Not Executable
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Out of Memory
```bash
# Use larger grid size to reduce memory usage
sed -i 's/cell_size_meters: .*/cell_size_meters: 250/' config/model_config.yaml
./run_pipeline.sh
```

### Config Not Found
```bash
# Verify config exists
ls -l config/model_config.yaml

# If missing, check you're in project root
pwd  # Should be: /path/to/uece_leonardo_trab_final
```

### Results Seem Wrong
```bash
# Check data quality
python -c "import pandas as pd; print(pd.read_parquet('data/processed/grids/cells.parquet').info())"

# Verify GTFS data exists
ls -lh data/raw/gtfs/
```

## 📚 Related Documentation

- **REGENERATION_GUIDE.md** - Detailed step-by-step instructions
- **GRID_SIZE_SOLUTION.md** - Why grid size matters and how to choose
- **QUICK_START_150m.txt** - Visual quick reference
- **reports/data_leakage_diagnostic.md** - Investigation of perfect model performance

## 💡 Tips

1. **Start with 150m** - Best balance between performance and computational cost
2. **Backup before big changes** - The script will prompt, but manual backup is safer
3. **Check results interpretation** - The script automatically interprets F1-scores
4. **Experiment systematically** - Try 150m → 200m → 250m to understand the tradeoff
5. **Use smaller grids for production** - More accurate for urban planning applications
6. **Use larger grids for prototyping** - Faster iterations during development

## 🎓 Academic Context

This pipeline was developed for a Machine Learning course project at UECE, focusing on:
- Binary classification of transit coverage in urban areas
- Spatial grid-based feature engineering
- Impact of spatial resolution on model performance
- Production-ready model deployment (ONNX format)

The discovery of perfect F1=1.0000 scores led to investigation of:
- Circular dependencies between features and labels
- Over-aggregation effects of coarse spatial grids
- The importance of choosing appropriate spatial scales

This demonstrates critical thinking and understanding of spatial analysis principles.

---

**Last Updated**: December 2024  
**Project**: Transit Coverage Classification - UECE ML Course  
**Script Version**: 2.0 (Generic Pipeline Execution)
