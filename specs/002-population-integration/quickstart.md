# Quick Start: IBGE Population Data Integration

**Feature**: 002-population-integration  
**Created**: 2025-12-10  
**Time to Complete**: ~15-20 minutes (first run with data download)

---

## Prerequisites

### Required Software
- Python 3.10 or higher
- Git (for cloning repository)
- 8 GB RAM minimum, 16 GB recommended
- 2 GB free disk space

### Required Data Files
- GTFS data: Already included in `data/raw/GTFSBHTRANS.zip`
- **IBGE data**: Must be downloaded separately (see step 1 below)

---

## Step 1: Download IBGE Population Data

### Option A: Direct Download (Recommended)

1. Visit IBGE Grade Estatística download page:
   ```
   https://www.ibge.gov.br/geociencias/downloads-geociencias.html
   ```

2. Navigate to: **Censo 2022 > Grade Estatística > Belo Horizonte**

3. Download the file: `grade_id36_belo_horizonte_2022.zip` (or similar)

4. Save to project directory:
   ```bash
   mv ~/Downloads/grade_id36_*.zip data/raw/ibge_populacao_bh_grade_id36.zip
   ```

### Option B: Using wget (if direct URL available)

```bash
cd data/raw/
wget https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Grade_Estatistica/Belo_Horizonte/grade_id36_bh_2022.zip \
     -O ibge_populacao_bh_grade_id36.zip
```

*Note: Replace URL with actual IBGE FTP link if available*

### Verify Download

```bash
# Check file exists and size is reasonable (5-20 MB)
ls -lh data/raw/ibge_populacao_bh_grade_id36.zip

# Verify ZIP integrity
unzip -t data/raw/ibge_populacao_bh_grade_id36.zip
```

Expected output: "No errors detected"

---

## Step 2: Update Configuration

Edit `config/model_config.yaml`:

```yaml
grid:
  cell_size_meters: 200  # Change from 250 to 200
  expected_cells: 17000  # Update from 13000

features:
  # Add this new section
  population:
    source_file: "data/raw/ibge_populacao_bh_grade_id36.zip"
    required_column: "POP"
    min_expected_population: 0
    max_expected_population: 3000
    total_population_range:
      min: 2000000
      max: 2800000
    coverage_threshold: 0.95
    max_zero_pop_pct: 0.10
    target_crs: "EPSG:4326"
```

**Quick edit commands**:
```bash
# Backup original config
cp config/model_config.yaml config/model_config_250m_backup.yaml

# Use sed for quick updates (or edit manually with text editor)
sed -i 's/cell_size_meters: 250/cell_size_meters: 200/' config/model_config.yaml
sed -i 's/expected_cells: 13000/expected_cells: 17000/' config/model_config.yaml
```

---

## Step 3: Install Dependencies

Ensure Python virtual environment is set up:

```bash
# Create virtual environment (if not already done)
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Install/update dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify GeoPandas installation**:
```bash
python -c "import geopandas; print(f'GeoPandas {geopandas.__version__}')"
```

Expected: `GeoPandas 0.14.x` or higher

---

## Step 4: Run Population Integration Pipeline

### Full Pipeline (Recommended for First Run)

This regenerates everything from scratch:

```bash
# Make script executable
chmod +x run_pipeline.sh

# Run full pipeline (takes ~8-10 minutes)
./run_pipeline.sh
```

**Pipeline steps**:
1. Grid generation (200m resolution) - ~60-90 seconds
2. GTFS data loading and processing - ~30 seconds
3. Transit feature extraction - ~5-7 minutes
4. **Population integration (NEW)** - ~10-20 seconds
5. Label generation - ~10 seconds
6. Model training - ~2-3 minutes
7. Model evaluation and export - ~30 seconds

### Incremental Pipeline (If Grid Already Exists)

If you've already generated the 200m grid and only want to add population:

```bash
# Step 4a: Integrate population with existing features
python src/data/population_integrator_cli.py \
    --grid-features data/processed/features/grid_features_transit_only.parquet \
    --ibge-zip data/raw/ibge_populacao_bh_grade_id36.zip \
    --output data/processed/features/grid_features.parquet

# Step 4b: Retrain model with population feature
python src/models/train.py \
    --features data/processed/features/grid_features.parquet \
    --config config/model_config.yaml \
    --output models/transit_coverage/
```

---

## Step 5: Verify Results

### Check Population Integration

```bash
# View population statistics
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features/grid_features.parquet')
print(f'Total cells: {len(df):,}')
print(f'Total population: {df[\"population\"].sum():,}')
print(f'Mean population: {df[df[\"population\"] > 0][\"population\"].mean():.1f}')
print(f'Coverage: {(df[\"population\"] > 0).mean():.1%}')
"
```

**Expected output**:
```
Total cells: 16,500-18,000
Total population: 2,300,000-2,500,000
Mean population: 60-400
Coverage: >95%
```

### Check Model Training

```bash
# View training summary
cat models/transit_coverage/training_summary.txt
```

Look for:
- ✅ Population listed in features
- ✅ Feature importance for population >5%
- ✅ Model accuracy metrics (accuracy, F1-score)

### Visualize Results (Optional)

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/02_feature_engineering.ipynb
```

Navigate to the population analysis section to see:
- Population distribution histogram
- Spatial map of population density
- Correlation between population and transit metrics

---

## Step 6: Generate Technical Report

Update the technical report to include population analysis:

```bash
python generate_report.py
```

**Generated files**:
- `reports/relatorio_tecnico.md` - Markdown report with population section
- `reports/relatorio_tecnico.pdf` - PDF report (requires LaTeX or pandoc)

**Key sections to review**:
- "Population Data Integration" - Rationale and methodology
- "Feature Importance Analysis" - Population's contribution to model
- "Model Performance Comparison" - Baseline vs enriched model metrics

---

## Troubleshooting

### Issue: IBGE file not found

**Error message**:
```
FileNotFoundError: IBGE data file not found: data/raw/ibge_populacao_bh_grade_id36.zip
```

**Solution**:
1. Verify file exists: `ls data/raw/ibge_populacao_bh_grade_id36.zip`
2. If missing, re-download from IBGE (see Step 1)
3. Check filename exactly matches config setting

---

### Issue: Missing POP column

**Error message**:
```
ValueError: IBGE data missing required 'POP' column.
Available columns: POPULACAO, ID36, geometry
```

**Solution**:
The IBGE file uses a different column name. Either:

**Option A**: Rename column in config (quick fix):
```yaml
features:
  population:
    required_column: "POPULACAO"  # Changed from "POP"
```

**Option B**: Preprocess file to rename column:
```python
import geopandas as gpd
gdf = gpd.read_file("zip://data/raw/ibge_populacao_bh_grade_id36.zip")
gdf = gdf.rename(columns={'POPULACAO': 'POP'})
gdf.to_file("data/raw/ibge_populacao_bh_grade_id36_fixed.zip", driver="ESRI Shapefile")
# Update config source_file to point to _fixed.zip
```

---

### Issue: Coverage below 95%

**Warning message**:
```
WARNING: Only 88.5% cells matched by ID36
```

**Solution**:
This indicates spatial mismatch between IBGE cells and generated grid. The system automatically falls back to spatial join:

1. Check that spatial join completed successfully
2. Verify final coverage rate after spatial join is >95%
3. If still below 95%, investigate unmatched cells:

```python
import geopandas as gpd
enriched = gpd.read_parquet("data/processed/features/grid_features.parquet")
unmatched = enriched[enriched['population'] == 0]
print(f"Unmatched cells: {len(unmatched)}")
# Visualize unmatched cells to see if they're edge cases
```

---

### Issue: Pipeline exceeds 10-minute limit

**Error message**:
```
ERROR: Pipeline exceeded 10-minute performance threshold
```

**Solution**:
1. **Reduce geographic bounds** (process smaller area):
   ```yaml
   grid:
     bounds:
       lat_min: -19.95  # Narrower bounds
       lat_max: -19.85
   ```

2. **Upgrade hardware**: Ensure 16 GB RAM, 4+ CPU cores

3. **Enable chunked processing** (future enhancement):
   ```yaml
   performance:
     enable_chunked_processing: true
     chunk_size: 5000
   ```

---

### Issue: Total population out of expected range

**Error message**:
```
ValueError: Total population 1,850,000 outside expected range [2,000,000, 2,800,000]
```

**Solution**:
This may indicate:
1. Geographic bounds don't fully cover Belo Horizonte → Expand bounds
2. IBGE data is for a different region → Verify data source
3. Expected range is misconfigured → Adjust config if BH population changed

**Verify population data**:
```python
import geopandas as gpd
ibge = gpd.read_file("zip://data/raw/ibge_populacao_bh_grade_id36.zip")
print(f"Total population in file: {ibge['POP'].sum():,}")
# Compare with known BH census figure
```

---

## Performance Benchmarks

Expected execution times on standard hardware (16 GB RAM, 4-core CPU):

| Step | Time |
|------|------|
| Grid generation (200m) | 60-90 sec |
| Transit feature extraction | 5-7 min |
| Population integration | 10-20 sec |
| Model training | 2-3 min |
| **Total pipeline** | **8-10 min** |

If your system significantly exceeds these times, consider:
- Closing other applications to free RAM
- Reducing grid size (smaller geographic bounds)
- Upgrading to SSD (if using HDD)

---

## Next Steps

After successful population integration:

1. **Compare model performance**: Review feature importance and metrics to assess population's impact

2. **Analyze case studies**: Use notebooks to explore how population context changes classifications

3. **Update documentation**: Share findings in README.md and technical reports

4. **API deployment** (optional): Deploy enriched model via FastAPI for real-time predictions

5. **Future enhancements**:
   - Add demographic segmentation (age, income)
   - Temporal population analysis (daytime vs nighttime)
   - Visualization dashboard with population overlays

---

## Quick Reference Commands

```bash
# Complete pipeline from scratch
./run_pipeline.sh

# Verify installation
python -c "import geopandas; import sklearn; print('OK')"

# Check data files
ls -lh data/raw/GTFSBHTRANS.zip data/raw/ibge_populacao_bh_grade_id36.zip

# View grid features
python -c "import pandas as pd; df = pd.read_parquet('data/processed/features/grid_features.parquet'); print(df.info())"

# Check model metrics
cat models/transit_coverage/training_summary.txt

# Generate report
python generate_report.py
```

---

## Support

- **Documentation**: See `reports/relatorio_tecnico.md` for detailed technical documentation
- **Spec**: Review `specs/002-population-integration/spec.md` for feature requirements
- **Research**: See `specs/002-population-integration/research.md` for design decisions
- **Issues**: Check error messages against troubleshooting section above

---

**Happy analyzing! 🚆📊🗺️**
