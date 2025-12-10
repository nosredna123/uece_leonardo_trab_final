# Trabalho Final - Aprendizado de Máquina e Mineração de Dados

**Disciplina:** Aprendizado de Máquina e Mineração de Dados 2025.2  
**Professor:** Leonardo Rocha

---

## 🚀 Quick Start Guide

### Initial Setup

1. **Run the setup script:**
   ```bash
   bash setup.sh
   ```
   This will:
   - Create a Python virtual environment in `.venv/`
   - Install all dependencies from `requirements.txt`

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

### 📂 Dataset

The project includes the **GTFSBHTRANS** (BH Trans GTFS) dataset, which is already located in `data/raw/GTFSBHTRANS.zip`. This dataset contains public transportation data for Belo Horizonte's transit system.

**Converting to Parquet:**

To convert the GTFS txt files to Parquet format (more efficient for processing):

```bash
python src/data/convert_to_parquet.py
```

This will extract and convert all txt files to Parquet format in `data/processed/gtfs/`.

**Using in your code:**

```python
from src.data.gtfs_loader import GTFSLoader

# Initialize loader
loader = GTFSLoader()

# Load all GTFS files
gtfs_data = loader.load_all_files()

# Or load a specific parquet file
df_stops = loader.load_parquet('stops')
df_routes = loader.load_parquet('routes')
```

### 📊 Development Workflow

#### Option A: Automated Pipeline (Recommended) ⚡

Run the complete ML pipeline with a single command:

```bash
# Run with current configuration
./run_pipeline.sh
```

This executes the full 8-step pipeline:
1. Spatial grid generation (based on `cell_size_meters` in config)
2. Feature extraction from GTFS data
3. Label generation for transit coverage
4. Data preprocessing (train/val/test splits)
5. Model training (Logistic Regression, Random Forest, Gradient Boosting)
6. Model evaluation and metrics
7. Model export to ONNX format
8. Report and visualization generation

**Customizing the Pipeline:**

Before running, edit `config/model_config.yaml` to adjust:

```bash
nano config/model_config.yaml
```

Key parameters:
- `cell_size_meters`: Grid resolution (100, 150, 200, 250, 500...)
  - **150m recommended** - best balance between accuracy and performance
- `test_size`: Test set proportion (default: 0.15)
- `validation_size`: Validation set proportion (default: 0.15)
- Model hyperparameters (max_iter, n_estimators, learning_rate, etc.)

**Expected Results:**
- Execution time: 5-10 minutes (150m grids), 2-3 minutes (250m grids)
- Output: F1-score ~0.75-0.85 (150m), ~0.85-0.92 (250m)
- Generated files: models, reports, visualizations

For detailed documentation, see:
- `PIPELINE_USAGE.md` - Comprehensive usage guide
- `REGENERATION_GUIDE.md` - Step-by-step instructions
- `GRID_SIZE_SOLUTION.md` - Grid size selection guide

#### Option B: Manual Step-by-Step (Advanced)

For exploratory analysis or custom workflows:

**Step 1: Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```
- Load and explore GTFS dataset
- Analyze transit coverage patterns
- Check data quality and distributions

**Step 2: Run Individual Pipeline Steps**
```bash
# Generate spatial grid
python -m src.data.grid_generator

# Extract features from GTFS data
python -m src.data.feature_extractor

# Generate labels
python -m src.data.label_generator

# Preprocess and split data
python -m src.data.preprocessing

# Train models
python -m src.models.train

# Evaluate models
python -m src.models.evaluator

# Export best model
python -m src.models.export

# Generate report
python generate_report.py
```

### 🤖 Running the API

After running the pipeline (which exports the model automatically):

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

#### Example API Request

Predict transit coverage for a specific location in Belo Horizonte:

```bash
# Check if a location has good transit coverage
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"latitude": -19.9167, "longitude": -43.9345}'

# Response:
# {
#   "prediction": 1,
#   "label": "well_served",
#   "probability": 0.89,
#   "features": {...}
# }
```

### 📝 Using Python Modules Directly

You can also use the modules directly in Python:

```python
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer

# Preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/your_data.csv')

# Train models
trainer = ModelTrainer()
trainer.initialize_models()
trainer.train_all_models(X_train, y_train, X_test, y_test)
```

### 🔧 Common Issues

#### Missing Dependencies
If you get import errors, make sure you:
1. Activated the virtual environment: `source .venv/bin/activate`
2. Installed all requirements: `pip install -r requirements.txt`

#### Script Not Executable
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

#### Model Not Found in API
Make sure you:
1. Ran the pipeline: `./run_pipeline.sh`
2. Check model exists: `ls -lh models/transit_coverage/best_model.onnx`
3. Restarted the API server

#### Out of Memory During Pipeline Execution
Use a larger grid size to reduce memory usage:
```bash
# Edit config to use 250m or 300m grids
nano config/model_config.yaml  # Set cell_size_meters: 250
./run_pipeline.sh
```

#### Pipeline Results Look Suspicious (F1 = 1.00)
This indicates over-aggregation. Use smaller grids:
```bash
# Edit config to use 150m or 200m grids
nano config/model_config.yaml  # Set cell_size_meters: 150
./run_pipeline.sh
```

See `reports/data_leakage_diagnostic.md` for detailed analysis.

---

## 🎯 Features

### Transit Coverage Classifier

**Status:** In Specification  
**Branch:** `1-transit-coverage-classifier`  
**Specification:** [specs/1-transit-coverage-classifier/spec.md](specs/1-transit-coverage-classifier/spec.md)

Binary classification model to identify well-served vs underserved regions in Belo Horizonte based on GTFS transit data. This feature supports urban planning decisions and equitable mobility policy analysis.

**Key Capabilities:**
- Geographic grid-based analysis of transit coverage
- Feature extraction from GTFS data (stops, routes, trip frequency)
- Binary classification: well-served (1) vs underserved (0)
- Model export to ONNX format
- API endpoint for real-time predictions

**Success Criteria:**
- F1-score ≥ 0.70 on test set
- API response time < 200ms per prediction
- Coverage analysis for 90%+ of city area

See the [full specification](specs/1-transit-coverage-classifier/spec.md) for details.

---

### 🎯 Next Steps

1. ✅ Setup environment (done by `setup.sh`)
2. ✅ Dataset ready in `data/raw/GTFSBHTRANS.zip`
3. ✅ Convert GTFS to Parquet: `python src/data/convert_to_parquet.py`
4. ⚙️ Configure pipeline: `nano config/model_config.yaml` (set `cell_size_meters: 150`)
5. 🚀 Run complete pipeline: `./run_pipeline.sh` (5-10 minutes)
6. 📊 Review results: `cat reports/tables/model_comparison.csv`
7. 📈 Check visualizations: `ls reports/figures/`
8. 🤖 Start API and test predictions: `uvicorn src.api.main:app --port 8000`
9. 📝 Read technical report: `reports/relatorio_tecnico.md`

**Optional:** Run exploratory analysis notebook for deeper insights:
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 💡 Tips

- Use `git add .` and `git commit` regularly to save progress
- Document your findings in the notebook markdown cells
- Export multiple model formats for compatibility
- Test the API thoroughly before final submission

---

## 📋 Descrição do Projeto

[Descreva aqui o problema abordado e a solução desenvolvida]

Este projeto implementa um pipeline completo de Machine Learning para [descrever a tarefa], utilizando o dataset [nome do dataset]. O objetivo é [descrever o objetivo principal].

## 🗂️ Estrutura do Repositório

```
.
├── data/                    # Dados do projeto
│   ├── raw/                # Dados brutos originais
│   └── processed/          # Dados processados
├── notebooks/              # Jupyter notebooks para análise exploratória
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                    # Código fonte do projeto
│   ├── __init__.py
│   ├── data/              # Scripts para tratamento de dados
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/          # Engenharia de features
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/            # Treinamento e exportação de modelos
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── export.py
│   └── api/               # API para servir modelos
│       ├── __init__.py
│       └── main.py
├── models/                 # Modelos treinados exportados
│   └── .gitkeep
├── tests/                  # Testes unitários
│   └── __init__.py
├── .gitignore             # Arquivos ignorados pelo Git
├── requirements.txt       # Dependências do projeto
├── setup.sh              # Script de setup do ambiente
├── README.md             # Este arquivo
└── trab-final-leonardo.pdf  # Especificação do trabalho

```

## 🚀 Como Executar

### 1. Configuração do Ambiente

Execute o script de setup para criar o ambiente virtual e instalar as dependências:

```bash
bash setup.sh
```

Ou manualmente:

```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar Notebooks

Com o ambiente ativado, inicie o Jupyter:

```bash
jupyter notebook
```

Navegue até a pasta `notebooks/` e execute os notebooks na ordem:
1. `01_exploratory_analysis.ipynb` - Análise exploratória dos dados
2. `02_feature_engineering.ipynb` - Engenharia de features
3. `03_model_training.ipynb` - Treinamento e validação de modelos

### 3. Treinar Modelos via Script

```bash
python src/models/train.py
```

### 4. Executar API de Model Serving

Inicie a API FastAPI:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse a documentação interativa em: `http://localhost:8000/docs`

#### Exemplo de Requisição

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [value1, value2, value3, ...]}'
```

## 📊 Dataset

**Nome:** GTFSBHTRANS - BH Trans GTFS Data  
**Fonte:** Dados de transporte público de Belo Horizonte  
**Localização:** `data/raw/GTFSBHTRANS.zip`  
**Formato:** GTFS (General Transit Feed Specification)  
**Tamanho:** ~213 MB (arquivo compactado)  
**Conteúdo:** 
- `stop_times.txt` - Horários de paradas
- `stops.txt` - Informações de paradas
- `routes.txt` - Rotas de ônibus
- `trips.txt` - Viagens
- `shapes.txt` - Geometrias das rotas
- `calendar.txt` e `calendar_dates.txt` - Calendários de operação
- Outros arquivos GTFS

**Tarefa:** [A ser definida - classificação, regressão, clustering, etc.]

## 🤖 Modelos Implementados

- **Modelo 1:** [Nome] - [Métricas principais]
- **Modelo 2:** [Nome] - [Métricas principais]
- **Modelo 3:** [Nome] - [Métricas principais]

**Melhor Modelo:** [Nome e justificativa]

## 📈 Resultados

[Incluir métricas principais, gráficos relevantes e análise dos resultados]

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Modelo 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| Modelo 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| Modelo 3 | 0.00 | 0.00 | 0.00 | 0.00 |

## 🔧 Tecnologias Utilizadas

- **Python 3.10+**
- **Scikit-learn** - Algoritmos de ML
- **Pandas/NumPy** - Manipulação de dados
- **Matplotlib/Seaborn/Plotly** - Visualização
- **FastAPI** - API REST
- **ONNX Runtime** - Model serving
- **Jupyter** - Notebooks interativos

## 📝 Dependências

Todas as dependências estão listadas em `requirements.txt`. Principais bibliotecas:
- numpy, pandas, scipy
- scikit-learn, xgboost, lightgbm
- onnx, onnxruntime
- fastapi, uvicorn
- jupyter, notebook

## 👥 Autor(es)

- [Seu Nome] - [Matrícula]
- [Nome do Parceiro] - [Matrícula] _(se aplicável)_

## 📄 Licença

Este projeto foi desenvolvido como trabalho acadêmico para a disciplina de Aprendizado de Máquina e Mineração de Dados da UECE.

## 🙏 Agradecimentos

- Prof. Leonardo Rocha
- [Outras referências ou agradecimentos]
