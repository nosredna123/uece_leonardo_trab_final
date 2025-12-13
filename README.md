# Trabalho Final - Aprendizado de Máquina e Mineração de Dados

**Disciplina:** Aprendizado de Máquina e Mineração de Dados - 2025.2  
**Professor:** Leonardo Rocha  
**Aluno:** Anderson Martins Gomes

---

## 📋 Descrição do Projeto

Este projeto implementa um **classificador de cobertura de transporte público** para a cidade de Belo Horizonte, utilizando dados GTFS (General Transit Feed Specification) do sistema BHTrans e dados populacionais do IBGE Censo 2022. O objetivo é identificar regiões bem atendidas versus regiões mal atendidas pelo transporte público, considerando tanto a oferta de serviço (paradas, rotas, frequência) quanto a demanda populacional.

### 🎯 Objetivo

Desenvolver um modelo de classificação binária que, dada uma localização geográfica (latitude/longitude) em Belo Horizonte, prediga se aquela região possui:
- **Classe 0 (Mal atendida):** Baixa oferta de transporte + Alta demanda populacional
- **Classe 1 (Bem atendida):** Alta oferta de transporte OU Baixa demanda populacional

### 💡 Motivação

A integração de dados populacionais do IBGE permite que o modelo compreenda o contexto de **demanda versus oferta**:
- Regiões com alta população e pouco transporte são **realmente mal atendidas** (prioridade para expansão)
- Regiões com baixa população e pouco transporte estão **adequadamente atendidas** (baixa demanda)
- Evita classificações enganosas baseadas apenas em métricas de transporte

---

## 🚀 Início Rápido

### 1. Configuração Inicial

Execute o script de setup para criar o ambiente virtual e instalar as dependências:

```bash
bash setup.sh
```

Isso irá:
- Criar ambiente virtual Python em `.venv/`
- Instalar todas as dependências do `requirements.txt`

### 2. Ativar Ambiente Virtual

```bash
source .venv/bin/activate
```

### 3. Executar Pipeline Completo

Execute o pipeline automatizado com um único comando:

```bash
./run_pipeline.sh
```

O pipeline executará **9 etapas** em sequência:

1. **Geração de Grid Espacial** (200m × 200m) - alinhado com dados IBGE
2. **Extração de Features** - métricas de transporte (paradas, rotas, viagens)
3. **Integração de Dados Populacionais** - IBGE Censo 2022
4. **Geração de Labels** - classificação baseada em oferta vs demanda
5. **Divisão dos Dados** - train/validation/test (70%/15%/15%)
6. **Treinamento de Modelos** - Logistic Regression, Random Forest, Gradient Boosting
7. **Avaliação de Modelos** - métricas e visualizações
8. **Exportação para ONNX** - modelo em formato portátil
9. **Geração de Relatório** - relatório técnico completo

**Tempo de Execução:** ~2-3 minutos  
**Configuração Utilizada:** Grid 200m, 20.125 células, 3.5M habitantes

---

## 📊 Resultados Obtidos

### Métricas dos Modelos

Utilizando grid de **200m × 200m** com integração de dados populacionais:

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tempo Treinamento |
|--------|----------|-----------|--------|----------|---------|-------------------|
| **Logistic Regression** | 0.8417 | 0.8103 | 0.9410 | 0.8707 | 0.8773 | 2.1s |
| **Random Forest** | **0.8831** | **0.8619** | **0.9451** | **0.9016** | **0.9008** | 42.8s |
| **Gradient Boosting** | 0.8811 | 0.8607 | 0.9427 | 0.8999 | 0.9035 | 44.5s |

🏆 **Melhor Modelo:** Random Forest
- **F1-Score no teste:** 0.9016
- **Accuracy:** 88.31%
- **Configuração:** 100 árvores, max_depth=10, min_samples_split=10

### Distribuição dos Dados

- **Total de células:** 20.125 (grid 200m × 200m)
- **População total:** 3.515.186 habitantes
- **Cobertura populacional:** 59.8% das células com dados do IBGE
- **Distribuição de classes:**
  - Mal atendidas (0): 8.719 células (43.3%)
  - Bem atendidas (1): 11.406 células (56.7%)

### Importância das Features

Baseado no modelo Random Forest:

| Feature | Importância | Descrição |
|---------|-------------|-----------|
| `daily_trips` | 35-40% | Frequência diária de viagens |
| `route_count` | 25-30% | Número de rotas de ônibus |
| `stop_count` | 20-25% | Número de paradas |
| `population` | 1.1% | Densidade populacional (IBGE) |
| Outras features | 10-15% | Geometria e features derivadas |

**Nota sobre população:** Embora a importância percentual seja pequena (1.1%), a feature populacional é **crítica** para distinguir entre áreas de baixa demanda (apropriadamente atendidas) e áreas de alta demanda mal atendidas.

### Visualizações Geradas

O pipeline gera automaticamente:

- **Matrizes de Confusão** - para cada modelo
- **Curvas ROC Comparativas** - desempenho de todos os modelos
- **Importância de Features** - análise comparativa
- **Relatório Técnico Completo** - `reports/relatorio_tecnico.md`

Arquivos disponíveis em:
- `reports/figures/*.png` - gráficos e visualizações
- `reports/tables/*.csv` - tabelas de métricas e comparações

---

## 📂 Dataset

### GTFS BHTrans

**Fonte:** Sistema de transporte público de Belo Horizonte  
**Localização:** `data/raw/GTFSBHTRANS.zip`  
**Formato:** GTFS (General Transit Feed Specification)  
**Tamanho:** ~213 MB compactado

**Arquivos principais:**
- `stops.txt` - 9.917 paradas de ônibus
- `routes.txt` - Rotas disponíveis
- `trips.txt` - Viagens programadas
- `stop_times.txt` - Horários em cada parada
- `shapes.txt` - Geometrias das rotas

**Conversão para Parquet:**
```bash
python src/data/convert_to_parquet.py
```

### IBGE Censo 2022

**Fonte:** Instituto Brasileiro de Geografia e Estatística  
**Localização:** `data/raw/ibge_populacao_bh_grade_id36.zip`  
**Formato:** Shapefile (Grade Estatística)  
**Resolução:** 200m × 200m (698.608 células)  
**População total:** 14.420.958 habitantes (região metropolitana)

**Integração:**
- Merge direto por ID de célula (alinhamento perfeito com grid 200m)
- 59.8% das células do grid contêm dados populacionais
- 3.5M habitantes na área de estudo (município de Belo Horizonte)

---

## 🏗️ Arquitetura da Pipeline

### Configuração (config/model_config.yaml)

```yaml
grid:
  cell_size_meters: 200  # Alinhado com IBGE
  bounds:
    min_lat: -20.08
    max_lat: -19.77
    min_lon: -44.08
    max_lon: -43.85

labeling:
  threshold_percentile: 90  # Top 10% = bem atendido
  weights:
    stops: 0.4
    routes: 0.3
    trips: 0.3
  noise:
    enabled: true
    population_noise_std: 0.25  # 25% variação
    threshold_noise_std: 0.15   # 15% variação
    label_flip_probability: 0.05  # 5% ruído nos labels

preprocessing:
  test_size: 0.15
  validation_size: 0.15
  random_state: 42
```

### Etapas da Pipeline

#### 1. Grid Espacial (`src/data/grid_generator.py`)
- Cria grid de 200m × 200m sobre Belo Horizonte
- Gera 20.125 células com geometria Polygon
- Calcula centroides e áreas
- Formato de saída: Parquet com CRS EPSG:4326

#### 2. Extração de Features (`src/data/feature_extractor.py`)
- Conta paradas por célula (média: 0.49)
- Conta rotas por célula (média: 0.82)
- Calcula frequência de viagens diárias (média: 72.4)
- Normaliza features com StandardScaler

#### 3. Integração Populacional (`src/data/population_integrator.py`)
- Carrega dados IBGE do Censo 2022
- Reprojecta para UTM Zone 23S (EPSG:31983) para cálculos geométricos
- Merge por ID de célula + fallback com spatial join
- Valida cobertura (≥50% requerido)

#### 4. Geração de Labels (`src/data/label_generator.py`)
- **Lógica oferta vs demanda:**
  - Calcula score composto de transporte (oferta)
  - Compara população com mediana (demanda)
  - **Mal atendido (0):** Baixa oferta AND Alta demanda
  - **Bem atendido (1):** Alta oferta OR Baixa demanda
- Adiciona ruído realístico (25% pop, 15% threshold, 5% flip)

#### 5. Pré-processamento (`src/data/preprocessing.py`)
- Divisão estratificada: 70% treino, 15% validação, 15% teste
- Preserva distribuição de classes em todos os splits
- Features normalizadas já na etapa 2

#### 6. Treinamento (`src/models/train.py`)
- Logistic Regression com GridSearchCV (4 combinações)
- Random Forest com RandomizedSearchCV (20 combinações)
- Gradient Boosting com RandomizedSearchCV (15 combinações)
- Validação cruzada 5-fold
- Seleção automática do melhor modelo (F1-score)

#### 7. Avaliação (`src/models/evaluator.py`)
- Calcula métricas no conjunto de teste
- Gera matrizes de confusão
- Plota curvas ROC comparativas
- Analisa importância de features
- Salva relatório de classificação

#### 8. Exportação ONNX (`src/models/export.py`)
- Converte melhor modelo para formato ONNX
- Valida predições (scikit-learn vs ONNX)
- Salva metadados (features, classes, métricas)
- Arquivo: `models/transit_coverage/best_model.onnx` (1.7 MB)

#### 9. Relatório (`generate_report.py`)
- Gera relatório técnico completo em Markdown
- Inclui metodologia, resultados, visualizações
- Estatísticas descritivas dos dados
- Arquivo: `reports/relatorio_tecnico.md`

---

## 🛠️ Uso Avançado

### Execução Passo a Passo

Para análise exploratória ou workflows customizados:

```bash
# 1. Gerar grid espacial
python -m src.data.grid_generator

# 2. Extrair features de transporte
python -m src.data.feature_extractor

# 3. Integrar dados populacionais
python -m src.data.population_integrator

# 4. Normalizar feature de população
python -m src.data.normalize_population

# 5. Gerar labels
python -m src.data.label_generator

# 6. Criar splits de dados
python -m src.data.preprocessing

# 7. Treinar modelos
python -m src.models.train

# 8. Avaliar modelos
python -m src.models.evaluator

# 9. Exportar para ONNX
python -m src.models.export

# 10. Gerar relatório
python generate_report.py
```

### Notebooks Jupyter

Para análise interativa:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Notebooks disponíveis:
- `01_exploratory_analysis.ipynb` - Análise exploratória dos dados GTFS
- `02_feature_engineering.ipynb` - Engenharia e análise de features
- `03_model_training.ipynb` - Treinamento e validação de modelos

### API REST

Após executar a pipeline, inicie a API para servir predições:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Documentação interativa:** http://localhost:8000/docs

**Exemplo de requisição:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"latitude": -19.9167, "longitude": -43.9345}'
```

**Resposta:**
```json
{
  "prediction": 1,
  "label": "well_served",
  "probability": 0.89,
  "features": {
    "stop_count": 3,
    "route_count": 5,
    "daily_trips": 245,
    "population": 1200
  }
}
```

---

## 🔧 Resolução de Problemas

### Dependências Faltando

Se encontrar erros de importação:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Script Não Executável

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Modelo Não Encontrado na API

Certifique-se de ter executado a pipeline:

```bash
./run_pipeline.sh
ls -lh models/transit_coverage/best_model.onnx
```

### Falta de Memória

Use grid maior (menor número de células):

```bash
nano config/model_config.yaml  # Alterar cell_size_meters: 250
./run_pipeline.sh
```

### Dados IBGE Não Encontrados

Se não tiver o arquivo `data/raw/ibge_populacao_bh_grade_id36.zip`:

1. A pipeline continuará sem integração populacional
2. Labels serão gerados apenas com métricas de transporte
3. Performance esperada: F1 ~0.97 (mas menos útil para planejamento urbano)

Para obter os dados IBGE:
- Acesse: https://www.ibge.gov.br/geociencias/downloads-geociencias.html
- Baixe: Grade Estatística Censo 2022 - Belo Horizonte
- Coloque em: `data/raw/ibge_populacao_bh_grade_id36.zip`

---

## 🗂️ Estrutura do Repositório

```
.
├── config/                      # Configurações
│   └── model_config.yaml       # Parâmetros do grid, features, modelos
├── data/                        # Dados do projeto
│   ├── raw/                    # Dados brutos
│   │   ├── GTFSBHTRANS.zip    # GTFS Belo Horizonte
│   │   └── ibge_populacao_bh_grade_id36.zip  # IBGE Censo 2022
│   └── processed/              # Dados processados
│       ├── grids/              # Grid espacial (Parquet)
│       ├── gtfs/               # GTFS convertido (Parquet)
│       ├── features/           # Features extraídas e splits
│       └── labels/             # Labels gerados
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                         # Código fonte
│   ├── data/                   # Processamento de dados
│   │   ├── grid_generator.py          # Geração de grid espacial
│   │   ├── gtfs_loader.py             # Carregamento GTFS
│   │   ├── feature_extractor.py       # Extração de features
│   │   ├── population_integrator.py   # Integração IBGE
│   │   ├── label_generator.py         # Geração de labels
│   │   └── preprocessing.py           # Splits e normalização
│   ├── models/                 # Treinamento e exportação
│   │   ├── train.py           # Treina LR, RF, GB
│   │   ├── evaluator.py       # Avaliação e visualizações
│   │   └── export.py          # Exportação ONNX
│   └── api/                   # API REST
│       ├── main.py            # FastAPI app
│       └── prediction_service.py
├── models/                     # Modelos treinados
│   └── transit_coverage/
│       ├── best_model.onnx            # Modelo exportado
│       ├── best_model.pkl             # Modelo scikit-learn
│       ├── model_metadata.json        # Metadados
│       └── training_summary.txt       # Resumo treinamento
├── reports/                    # Relatórios e visualizações
│   ├── figures/               # Gráficos (PNG)
│   │   ├── confusion_matrix_*.png
│   │   ├── roc_curves_comparison.png
│   │   └── feature_importance_comparison.png
│   ├── tables/                # Tabelas (CSV)
│   │   ├── model_comparison.csv
│   │   ├── feature_importance.csv
│   │   └── classification_report.txt
│   └── relatorio_tecnico.md   # Relatório completo
├── specs/                      # Especificações técnicas
│   ├── 001-transit-coverage-classifier/
│   └── 002-population-integration/
├── tests/                      # Testes
│   ├── unit/                  # Testes unitários
│   └── integration/           # Testes de integração
├── run_pipeline.sh            # Script automatizado (9 etapas)
├── setup.sh                   # Setup do ambiente
├── requirements.txt           # Dependências Python
└── README.md                  # Este arquivo
```

## 🔧 Tecnologias Utilizadas

### Core
- **Python 3.12+** - Linguagem principal
- **Scikit-learn 1.3+** - Algoritmos de ML (Logistic Regression, Random Forest, Gradient Boosting)
- **Pandas 2.1+** - Manipulação e análise de dados tabulares
- **NumPy 1.26+** - Computação numérica e álgebra linear
- **GeoPandas 0.14+** - Análise espacial e operações geométricas

### Visualização
- **Matplotlib 3.8+** - Gráficos e visualizações
- **Seaborn 0.13+** - Visualizações estatísticas

### Machine Learning
- **ONNX Runtime 1.16+** - Inferência de modelos em produção
- **scikit-learn** - Algoritmos, pré-processamento, validação cruzada

### API e Deployment
- **FastAPI 0.104+** - Framework web assíncrono
- **Uvicorn 0.24+** - ASGI server
- **Pydantic 2.5+** - Validação de dados

### Geoespacial
- **Shapely 2.0+** - Operações geométricas
- **PyProj 3.6+** - Transformações de coordenadas e projeções
- **Fiona 1.9+** - Leitura/escrita de dados geoespaciais

### Notebooks e Análise
- **Jupyter 1.0+** - Ambiente interativo
- **IPython 8.18+** - Shell interativo
- **ipykernel 6.27+** - Kernel Jupyter

## 📦 Dependências Completas

Arquivo `requirements.txt` com todas as dependências:

```txt
numpy>=1.26.0
pandas>=2.1.0
scipy>=1.11.0
scikit-learn>=1.3.0
geopandas>=0.14.0
shapely>=2.0.0
pyproj>=3.6.0
fiona>=1.9.0
onnx>=1.15.0
onnxruntime>=1.16.0
skl2onnx>=1.16.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
pyyaml>=6.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
ipython>=8.18.0
ipykernel>=6.27.0
tqdm>=4.66.0
requests>=2.31.0
```

**Instalação:**
```bash
pip install -r requirements.txt
```

## 👥 Autor

**Anderson Martins Gomes**  
Universidade Estadual do Ceará (UECE)  
Disciplina: Aprendizado de Máquina e Mineração de Dados - 2025.2  
Professor: Leonardo Rocha

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como trabalho final da disciplina de Aprendizado de Máquina e Mineração de Dados da UECE, sob orientação do Prof. Leonardo Rocha.

### Objetivos da Disciplina Atingidos

✅ **Pré-processamento de dados geoespaciais** - Grid, features, normalização  
✅ **Engenharia de features** - Extração de métricas de transporte e população  
✅ **Treinamento de múltiplos modelos** - LR, RF, GB com hyperparameter tuning  
✅ **Validação cruzada** - 5-fold CV com busca de hiperparâmetros  
✅ **Avaliação de modelos** - Métricas, visualizações, análise comparativa  
✅ **Exportação para produção** - Formato ONNX para deployment  
✅ **API REST** - Endpoint para predições em tempo real  
✅ **Documentação técnica** - Relatório completo com metodologia e resultados

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos. Os dados utilizados são de domínio público (GTFS BHTrans e IBGE Censo 2022).

---

## 📧 Contato

Para dúvidas ou sugestões sobre este projeto, entre em contato através dos canais da disciplina ou abra uma issue no repositório.

---

**Última atualização:** Dezembro 2025  
**Versão:** 2.0 (com integração populacional IBGE)
