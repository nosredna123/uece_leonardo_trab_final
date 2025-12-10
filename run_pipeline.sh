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

# Backup (optional)
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
read -p "⚠️  Backup existing results to ${BACKUP_DIR}/? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating backup..."
    mkdir -p "$BACKUP_DIR"
    cp -r data/processed "$BACKUP_DIR/" 2>/dev/null || true
    cp -r models "$BACKUP_DIR/" 2>/dev/null || true
    cp -r reports "$BACKUP_DIR/" 2>/dev/null || true
    echo "✓ Backup saved to: $BACKUP_DIR"
    echo ""
fi

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

# Step 1: Generate grid
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/8: Generating ${GRID_SIZE}m × ${GRID_SIZE}m spatial grid..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.data.grid_generator
echo "✓ Grid generated"
echo ""

# Step 2: Extract features
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/8: Extracting transit coverage features..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.data.feature_extractor
echo "✓ Features extracted"
echo ""

# Step 3: Generate labels
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/8: Generating binary labels..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.data.label_generator
echo "✓ Labels generated"
echo ""

# Step 4: Prepare splits
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4/8: Creating train/val/test splits..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.data.preprocessing
echo "✓ Data splits created"
echo ""

# Step 5: Train models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5/8: Training models (LR, RF, GB)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.models.train
echo "✓ Models trained"
echo ""

# Step 6: Evaluate models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6/8: Evaluating models and generating plots..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.models.evaluator
echo "✓ Evaluation complete"
echo ""

# Step 7: Export to ONNX
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 7/8: Exporting best model to ONNX..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m src.models.export
echo "✓ Model exported"
echo ""

# Step 8: Regenerate report
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 8/8: Regenerating technical report..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python generate_report.py
echo "✓ Report generated"
echo ""

# Display results summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ PIPELINE EXECUTION COMPLETE!                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
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
