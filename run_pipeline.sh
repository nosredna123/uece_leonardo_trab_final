#!/bin/bash
# Transit Coverage Pipeline - Full Execution Script
# 
# This script runs the complete ML pipeline from scratch:
#   1. Spatial grid generation
#   2. Feature extraction
#   3. Label generation
#   4. Data preprocessing
#   5. Model training
#   6. Model evaluation
#   7. Model export (ONNX)
#   8. Report generation
# 
# Usage:
#   ./run_pipeline.sh
#
# Configuration:
#   Edit config/model_config.yaml before running to adjust:
#   - cell_size_meters (grid resolution: 100, 150, 200, 250, 500...)
#   - model hyperparameters
#   - train/val/test split ratios

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Transit Coverage Pipeline - Full Execution                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 0: Stop any running API/uvicorn processes
echo "🛑 Stopping any running API processes..."
pkill -f "uvicorn.*src.api.main" 2>/dev/null || true
sleep 1
echo "✓ Running processes stopped"
echo ""

# Read current grid size from config
GRID_SIZE=$(grep "cell_size_meters:" config/model_config.yaml | awk '{print $2}')

if [ -z "$GRID_SIZE" ]; then
    echo "❌ ERROR: Could not read cell_size_meters from config/model_config.yaml"
    exit 1
fi

echo "✓ Config verified: cell_size_meters = ${GRID_SIZE}m"
echo ""

# Calculate expected cells (approximate)
AREA_KM2=812  # Approximate coverage area for Belo Horizonte
CELL_AREA=$(echo "scale=4; ($GRID_SIZE / 1000) ^ 2" | bc)
EXPECTED_CELLS=$(echo "scale=0; $AREA_KM2 / $CELL_AREA" | bc)

echo "📊 Grid Configuration:"
echo "   • Cell size: ${GRID_SIZE}m × ${GRID_SIZE}m"
echo "   • Cell area: ${CELL_AREA} km²"
echo "   • Expected cells: ~$(printf "%'d" $EXPECTED_CELLS)"
echo ""

# Clean previous pipeline outputs
echo "🧹 Cleaning previous pipeline outputs..."
echo "   • Removing grid data..."
rm -rf data/processed/grids/*
echo "   • Removing features..."
rm -rf data/processed/features/*
echo "   • Removing labels..."
rm -rf data/processed/labels/*
echo "   • Removing trained models (forces retraining)..."
rm -f models/transit_coverage/*.pkl
rm -f models/transit_coverage/*.onnx
rm -f models/transit_coverage/*.json
echo "   • Removing reports and visualizations..."
rm -rf reports/figures/*.png
rm -rf reports/tables/*.txt
rm -rf reports/tables/*.csv
echo "✓ All pipeline outputs cleaned (ready for fresh regeneration)"
echo ""
echo "🚀 Starting pipeline execution..."
echo ""

# Initialize timing
PIPELINE_START=$(date +%s)
POPULATION_TIME=0

# Step 1: Generate grid
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/9: Generating ${GRID_SIZE}m × ${GRID_SIZE}m spatial grid..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.data.grid_generator
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Grid generated (${STEP_TIME}s)"
echo ""

# Step 2: Extract features
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/9: Extracting transit coverage features..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.data.feature_extractor
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Features extracted (${STEP_TIME}s)"
echo ""

# Step 2.5: Integrate population data (NEW)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/9: Integrating IBGE population data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
# Check if IBGE data file exists before attempting integration
IBGE_ZIP="data/raw/ibge_populacao_bh_grade_id36.zip"
if [ -f "$IBGE_ZIP" ]; then
    echo "✓ IBGE data found: $IBGE_ZIP"
    python -m src.data.population_integrator
    # Rename output to replace the original features file
    if [ -f "data/processed/features/grid_features_with_population.parquet" ]; then
        mv data/processed/features/grid_features.parquet data/processed/features/grid_features_transit_only.parquet
        mv data/processed/features/grid_features_with_population.parquet data/processed/features/grid_features.parquet
        echo "✓ Population data integrated successfully"
        
        # Normalize population feature
        echo "  Normalizing population feature..."
        python -m src.data.normalize_population
        echo "✓ Population feature normalized"
    else
        echo "⚠️  Population integration failed - continuing with transit-only features"
    fi
else
    echo "⚠️  IBGE data not found at: $IBGE_ZIP"
    echo "    Skipping population integration (will use transit-only features)"
    echo "    To enable: Download IBGE data and place at $IBGE_ZIP"
fi
STEP_END=$(date +%s)
POPULATION_TIME=$((STEP_END - STEP_START))
echo "✓ Population integration completed (${POPULATION_TIME}s)"
if [ $POPULATION_TIME -gt 300 ]; then
    echo "⚠️  WARNING: Population integration exceeded 5-minute limit (FR-017)"
fi
echo ""

# Step 3: Generate labels
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4/9: Generating binary labels..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.data.label_generator
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Labels generated (${STEP_TIME}s)"
echo ""

# Step 4: Prepare splits
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5/9: Creating train/val/test splits..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.data.preprocessing
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Data splits created (${STEP_TIME}s)"
echo ""

# Step 5: Train models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6/9: Training models (LR, RF, GB)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.models.train
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Models trained (${STEP_TIME}s)"
echo ""

# Step 6: Evaluate models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 7/9: Evaluating models and generating plots..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.models.evaluator
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Evaluation complete (${STEP_TIME}s)"
echo ""

# Step 7: Export to ONNX
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 8/9: Exporting best model to ONNX..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python -m src.models.export
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Model exported (${STEP_TIME}s)"
echo ""

# Step 8: Regenerate report
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 9/9: Regenerating technical report..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_START=$(date +%s)
python generate_report.py
STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
echo "✓ Report generated (${STEP_TIME}s)"
echo ""

# Calculate total execution time
PIPELINE_END=$(date +%s)
TOTAL_TIME=$((PIPELINE_END - PIPELINE_START))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_SEC=$((TOTAL_TIME % 60))

# Display results summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ PIPELINE EXECUTION COMPLETE!                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Execution Time:"
echo "   Total:        ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "   Population:   ${POPULATION_TIME}s"
if [ $TOTAL_TIME -gt 600 ]; then
    echo "   ⚠️  WARNING: Total execution exceeded 10-minute target (FR-017)"
fi
if [ $POPULATION_TIME -gt 300 ]; then
    echo "   ⚠️  WARNING: Population integration exceeded 5-minute target (FR-017)"
fi
echo ""

# Show comparison
python -c "
import pandas as pd
import sys

print('📊 RESULTS SUMMARY')
print('=' * 70)

# Read grid size from environment
grid_size = ${GRID_SIZE}

# New results
try:
    new = pd.read_csv('reports/tables/model_comparison.csv')
    
    # Find best model by F1 score
    best_idx = new['f1_score'].idxmax()
    best = new.iloc[best_idx]
    
    print(f'\n{grid_size}m Grid Results:')
    print(f'  Best Model: {best[\"model_name\"]}')
    print(f'  F1-Score:   {best[\"f1_score\"]:.4f}')
    print(f'  Accuracy:   {best[\"accuracy\"]:.4f}')
    
    # Provide interpretation based on F1 score
    f1 = float(best['f1_score'])
    if f1 >= 0.98:
        print('  Status:     ⚠️  Very high - check for data leakage or over-aggregation')
    elif f1 >= 0.90:
        print('  Status:     ✅ Excellent performance')
    elif f1 >= 0.80:
        print('  Status:     ✅ Good performance - realistic for spatial classification')
    elif f1 >= 0.70:
        print('  Status:     ✓ Acceptable performance')
    else:
        print('  Status:     ⚠️  Low performance - consider larger grids or feature engineering')
    
    print('')
    print('All Models:')
    print('-' * 70)
    for idx, row in new.iterrows():
        print(f'  {row[\"model_name\"]:<20} F1={row[\"f1_score\"]:.4f}  Acc={row[\"accuracy\"]:.4f}')
    
    print('\n' + '=' * 70)
    print(f'✨ Pipeline executed successfully with {grid_size}m spatial resolution!')
    print('=' * 70)
except Exception as e:
    print(f'⚠️  Could not read results: {e}')
    print('Check: reports/tables/model_comparison.csv')
    sys.exit(1)
"

echo ""
echo "📁 Generated Files:"
echo "   • data/processed/grids/cells.parquet"
echo "   • data/processed/features/*.parquet"
echo "   • models/transit_coverage/best_model.onnx"
echo "   • reports/figures/*.png"
echo "   • reports/tables/*.csv"
echo "   • reports/relatorio_tecnico.md"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review results: cat reports/tables/model_comparison.csv"
echo "   2. Check visualizations: ls reports/figures/"
echo "   3. View report: head -100 reports/relatorio_tecnico.md"
echo ""
echo "🚀 Starting API server with auto-reload..."
echo ""

# Kill any remaining processes on port 8000
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1

# Start API with reload
echo "Starting uvicorn on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uvicorn src.api.main:app --reload --port 8000
