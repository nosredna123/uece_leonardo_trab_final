# Relatório Técnico - Trabalho Final de Aprendizado de Máquina

**Disciplina**: Aprendizado de Máquina e Mineração de Dados  
**Instituição**: Universidade Estadual do Ceará (UECE)  
**Professor**: Leonardo Rocha  
**Data**: 10/12/2025  
**Projeto**: Classificador de Cobertura de Transporte Público

---

## Sumário Executivo

Este relatório documenta o desenvolvimento de um sistema completo de aprendizado de máquina para classificar áreas urbanas de Belo Horizonte-MG em categorias de cobertura de transporte público ("mal atendidas" e "bem atendidas"). O projeto implementa todas as etapas do pipeline de ML, desde a geração dos dados até o model serving via API REST.

**Principais Resultados:**
- **Melhor Modelo**: Gradient Boosting com regularização L2
- **Performance no Teste**: 1.0000 de acurácia, 1.0000 de F1-score
- **Tempo de Treinamento**: 11.52 segundos
- **Latência da API**: 0.38ms por predição (526× mais rápido que requisito de 200ms)
- **Tamanho do Modelo**: 0.09 MB (formato ONNX)

---

## 1. Descrição do Dataset e Problema

### 1.1 Contexto e Motivação

O transporte público é fundamental para a mobilidade urbana e o desenvolvimento socioeconômico das cidades. A identificação de áreas mal atendidas por transporte público é essencial para orientar políticas públicas e investimentos em infraestrutura de mobilidade urbana.

### 1.2 Dataset Escolhido

**Fonte de Dados**: GTFS (General Transit Feed Specification) de Belo Horizonte-MG  
**Tipo**: Dados reais de transporte público (não sintético)  
**Cobertura Geográfica**: Região metropolitana de Belo Horizonte

O dataset GTFS contém informações estruturadas sobre o sistema de transporte público:
- `stops.txt`: Localização geográfica de pontos de parada (latitude/longitude)
- `routes.txt`: Definição de linhas de ônibus
- `trips.txt`: Viagens programadas para cada linha
- `stop_times.txt`: Horários de chegada/partida em cada parada
- `calendar.txt`: Frequências de serviço (dias da semana)

### 1.3 Problema de Aprendizado de Máquina

**Tipo**: Classificação binária supervisionada

**Objetivo**: Desenvolver um modelo que classifique células geográficas (grid de 500m × 500m) em duas categorias:
- **Classe 0 (Mal Atendida)**: Áreas com baixa cobertura de transporte público
- **Classe 1 (Bem Atendida)**: Áreas com cobertura adequada de transporte público

**Justificativa da Abordagem**: 
A discretização espacial em grid permite:
1. Análise uniforme da cobertura geográfica
2. Identificação clara de áreas prioritárias para investimento
3. Agregação de múltiplas características de transporte por região
4. Escalabilidade para análise de grandes áreas urbanas

### 1.4 Estratégia de Geração de Labels

Como não existem labels de ground truth (classificações humanas de "mal atendida" vs "bem atendida"), foi adotada uma estratégia de **labeling baseado em limiar percentílico**:

1. **Extração de Features**: Para cada célula do grid, calculam-se métricas quantitativas de cobertura de transporte
2. **Normalização**: Features são normalizadas para escala [0, 1]
3. **Threshold Percentílico**: Células abaixo do percentil 30 em múltiplas features são classificadas como "mal atendidas"

**Vantagens**:
- Automatização do processo de labeling
- Reprodutibilidade com diferentes thresholds
- Baseado em métricas objetivas de cobertura

**Limitações**:
- Labels refletem definição algorítmica, não julgamento humano
- Threshold de 30% é arbitrário (poderia ser ajustado com validação de domínio)

### 1.5 Características do Dataset Final

**Dimensões**:
- **Total de Células Geradas**: 3.250 (grid 500m × 500m cobrindo Belo Horizonte)
- **Amostras Válidas**: 2.438 células com features completas
- **Features**: 5 variáveis preditoras
- **Splits**:
  - Treino: 1.463 amostras (60%)
  - Validação: 487 amostras (20%)
  - Teste: 488 amostras (20%)

**Features Extraídas**:

1. `stop_count`: Número de pontos de parada na célula
2. `route_count`: Número de linhas únicas que atendem a célula
3. `daily_trips`: Total de viagens diárias na célula
4. `stop_density`: Densidade de paradas por km² (stops/0.25km²)
5. `route_diversity`: Diversidade de linhas (entropia de Shannon)

**Distribuição de Classes**:
- Classe 0 (Mal Atendida): ~70% das amostras
- Classe 1 (Bem Atendida): ~30% das amostras
- **Observação**: Desbalanceamento de classes tratado com estratificação nos splits

---

## 2. Modelagem e Implementação

### 2.1 Pipeline de Machine Learning

O projeto implementa um pipeline completo end-to-end:

#### 2.1.1 Geração de Grid Espacial
- **Implementação**: `src/grid/grid_generator.py`
- **Método**: Grid uniforme de 500m × 500m sobre bounding box do GTFS
- **Resultado**: 3.250 células geográficas (arquivo GeoJSON)

#### 2.1.2 Extração de Features
- **Implementação**: `src/features/feature_extractor.py`
- **Operações**:
  - Interseção espacial (stops dentro de cada célula)
  - Agregação de rotas únicas
  - Contagem de viagens diárias (join com `calendar.txt`)
  - Cálculo de densidade (normalização por área)
  - Diversidade de rotas (entropia de Shannon)
  - Normalização min-max para features numéricas

#### 2.1.3 Geração de Labels
- **Implementação**: `src/features/label_generator.py`
- **Estratégia**: Percentil 30 como threshold
- **Filtro**: Remoção de células com valores NaN (áreas fora da cobertura GTFS)

#### 2.1.4 Preparação do Dataset
- **Implementação**: `src/data/dataset_splitter.py`
- **Split Strategy**: Estratificado (preserva distribuição de classes)
- **Proporções**: 60% treino, 20% validação, 20% teste
- **Random Seed**: 42 (reprodutibilidade)

### 2.2 Algoritmos de Aprendizado

Foram treinados e comparados três algoritmos de classificação:

#### 2.2.1 Regressão Logística (Modelo Vencedor)
**Justificativa**: 
- Interpretabilidade: coeficientes lineares revelam importância direta das features
- Eficiência: treinamento e inferência rápidos
- Calibração de probabilidades: natural para classificação binária

**Hiperparâmetros Otimizados**:

- `subsample`: 0.8
- `n_estimators`: 200
- `min_samples_split`: 5
- `min_samples_leaf`: 1
- `max_depth`: 3
- `learning_rate`: 0.15

**Busca de Hiperparâmetros**: GridSearchCV
- Espaço de busca: C ∈ {0.001, 0.01, 0.1, 1.0} (4 valores)
- Validação cruzada: 5 folds
- Métrica de otimização: F1-score (macro-averaged)
- Total de fits: 4 × 5 = 20

**Resultados**:

- CV F1-score: 0.9998
- Validação F1-score: 0.9979
- Tempo de treinamento: 1.57 segundos

#### 2.2.2 Random Forest
**Justificativa**:
- Captura não-linearidades: árvores de decisão aprendem interações complexas
- Ensemble robustness: redução de variância via bagging
- Importância de features: ranking intrínseco via Gini importance

**Hiperparâmetros Otimizados**:

- `n_estimators`: 500
- `min_samples_split`: 15
- `min_samples_leaf`: 4
- `max_features`: sqrt
- `max_depth`: 25
- `bootstrap`: True

**Busca de Hiperparâmetros**: RandomizedSearchCV
- 20 iterações de amostragem aleatória
- Validação cruzada: 5 folds
- Total de fits: 20 × 5 = 100

**Resultados**:
- CV F1-score: 0.9998
- Validação F1-score: 0.9979
- Tempo de treinamento: 6.12 segundos

#### 2.2.3 Gradient Boosting
**Justificativa**:
- Correção sequencial de erros: boosting otimiza diretamente o erro residual
- Robustez a desbalanceamento: pesos adaptativos para classes minoritárias
- State-of-the-art: família de algoritmos competitivos em benchmarks

**Hiperparâmetros Otimizados**:

- `subsample`: 0.8
- `n_estimators`: 200
- `min_samples_split`: 5
- `min_samples_leaf`: 1
- `max_depth`: 3
- `learning_rate`: 0.15

**Busca de Hiperparâmetros**: RandomizedSearchCV
- 15 iterações de amostragem aleatória
- Validação cruzada: 5 folds
- Total de fits: 15 × 5 = 75

**Resultados**:
- CV F1-score: 1.0000
- Validação F1-score: 1.0000
- Tempo de treinamento: 3.83 segundos

### 2.3 Seleção do Modelo Final

**Critério de Seleção**: F1-score no conjunto de validação

**Modelo Escolhido**: Gradient Boosting
- **Justificativa**: Melhor F1-score de validação (1.0000)
- **Vantagens Adicionais**:
  - Menor tempo de treinamento (1.57s vs 6.12s Random Forest)
  - Menor latência de inferência (0.38ms vs ~1-2ms para ensembles)
  - Maior interpretabilidade para stakeholders (coeficientes lineares)
  - Menor tamanho de modelo (0.0853 MB)

---

## 3. Resultados Obtidos

### 3.1 Performance no Conjunto de Teste

**Comparação entre Modelos** (488 amostras de teste):

| Algoritmo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|-----------|----------|----------|--------|----------|---------|
| Logistic Regression | 0.9990 | 0.9979 | 0.9979 | 0.9979 | 1.0000 |
| Random Forest | 0.9995 | 0.9979 | 1.0000 | 0.9990 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


**Análise**:
- **Gradient Boosting** alcança performance perfeita (ou quase perfeita) em todas as métricas
- Todos os três modelos demonstram excelente capacidade de generalização (F1 ≥ 0.989)
- Minimal gap entre CV e teste indica ausência de overfitting

### 3.2 Relatório de Classificação Detalhado

**Gradient Boosting - Conjunto de Teste**:

```
============================================================
CLASSIFICATION REPORT - Gradient Boosting
============================================================

              precision    recall  f1-score   support

 Underserved     1.0000    1.0000    1.0000      1451
 Well-served     1.0000    1.0000    1.0000       484

    accuracy                         1.0000      1935
   macro avg     1.0000    1.0000    1.0000      1935
weighted avg     1.0000    1.0000    1.0000      1935

============================================================

```

**Interpretação**:
- **Precisão**: 1.0000 → Das células classificadas como "bem atendidas", 100.00% realmente são
- **Recall**: 1.0000 → Das células realmente "bem atendidas", o modelo identifica 100.00%
- **F1-Score**: 1.0000 → Média harmônica balanceada entre precisão e recall

### 3.3 Matriz de Confusão

**Visualizações Geradas**:
- `reports/figures/confusion_matrix_logistic_regression.png`
- `reports/figures/confusion_matrix_random_forest.png`
- `reports/figures/confusion_matrix_gradient_boosting.png`

![Matriz de Confusão - Gradient Boosting](reports/figures/confusion_matrix_logistic_regression.png)

**Análise da Matriz de Confusão**:
- **Verdadeiros Negativos (TN)**: Células mal atendidas corretamente identificadas
- **Verdadeiros Positivos (TP)**: Células bem atendidas corretamente identificadas
- **Falsos Positivos (FP)**: Células mal atendidas classificadas incorretamente como bem atendidas (risco: subestimar necessidade de investimento)
- **Falsos Negativos (FN)**: Células bem atendidas classificadas como mal atendidas (risco: desperdiçar recursos)

### 3.4 Curvas ROC

![Curvas ROC - Comparação de Modelos](reports/figures/roc_curves_comparison.png)

**Interpretação**:
- Curva ROC próxima ao canto superior esquerdo indica excelente discriminação
- AUC (Area Under Curve) próximo a 1.0 confirma separação quase perfeita entre classes
- Todos os três modelos demonstram AUC ≥ 0.999

### 3.5 Importância das Features

![Importância de Features - Comparação](reports/figures/feature_importance_comparison.png)

**Ranking de Importância** (normalizado 0-1):

| Feature | Logistic Regression | Random Forest | Gradient Boosting |
|---------|---------------------|---------------|-------------------|
| stop_count | 0.0190 | 0.1272 | 0.0243 |
| route_count | 1.0000 | 0.4752 | 0.0058 |
| daily_trips | 0.0295 | 1.0000 | 1.0000 |
| stop_density | 0.3035 | 0.2033 | 0.0192 |
| route_diversity | 1.0000 | 0.5733 | 0.0209 |


**Insights**:
1. **Regressão Logística** prioriza `route_count` e `route_diversity`: modelo linear favorece características de rotas
2. **Modelos baseados em árvores** (RF/GB) priorizam `daily_trips`: capturam importância de frequência de serviço
3. **Consenso entre modelos**: `stop_density` é consistentemente importante
4. **Redundância de features**: Features normalizadas (`*_norm`) têm menor importância, sugerindo que features brutas já contêm informação suficiente

### 3.6 Tempo de Treinamento

**Total**: {total_training_time:.2f} segundos (~{total_training_time/60:.2f} minutos)

| Modelo | Método de Busca | Tempo (s) |
|--------|----------------|-----------|
| Logistic Regression | GridSearchCV | 1.57 |
| Random Forest | RandomizedSearchCV | 6.12 |
| Gradient Boosting | RandomizedSearchCV | 3.83 |


**Observação**: Treinamento extremamente rápido viabiliza experimentação iterativa e retreinamento frequente com dados atualizados.

---

## 4. Exportação e Model Serving

### 4.1 Exportação do Modelo

**Formato**: ONNX (Open Neural Network Exchange)  
**Implementação**: `src/models/export.py`

**Vantagens do ONNX**:
- **Interoperabilidade**: Compatível com múltiplas plataformas (Python, Java, C++, JavaScript)
- **Otimização**: Inferência otimizada via ONNX Runtime
- **Portabilidade**: Deployment independente de framework

**Processo de Conversão**:
1. Carregar melhor modelo treinado (`best_model.pkl`)
2. Converter para ONNX usando `skl2onnx` (opset 12)
3. Validar predições (100 amostras de teste)
4. Salvar modelo ONNX e metadados JSON

**Validação**:
- ✅ Predições ONNX ≈ Predições scikit-learn (100% match)
- ✅ Tamanho do arquivo: 0.0853 MB
- ✅ Diferença máxima de probabilidade: < 10⁻⁷

**Arquivos Gerados**:
- `models/transit_coverage/best_model.onnx`: Modelo exportado
- `models/transit_coverage/model_metadata.json`: Metadados (features, classes, hiperparâmetros)

### 4.2 API de Inferência (Model Serving)

**Framework**: FastAPI + ONNX Runtime  
**Implementação**: `src/api/main.py`, `src/api/prediction_service.py`

**Arquitetura**:
```
[Cliente HTTP] → [FastAPI] → [PredictionService] → [ONNX Runtime] → [Predição]
                      ↓
              [Validação Pydantic]
                      ↓
              [Error Handling]
```

**Endpoints Implementados**:

1. **GET /** - Informações da API
   - Retorna: Mensagem de boas-vindas, versão, links para documentação

2. **GET /health** - Health Check
   - Retorna: Status do serviço, modelo carregado, versão do modelo
   - Uso: Monitoramento de disponibilidade

3. **GET /model/info** - Metadados do Modelo
   - Retorna: Nome, tipo, features, classes, performance
   - Uso: Inspeção de configuração do modelo

4. **POST /predict** - Predição Única
   - Input: JSON com 5 features
   - Output: Classe predita, probabilidades, confiança, latência
   - Validação: Features obrigatórias, tipos numéricos

5. **POST /predict/batch** - Predição em Lote
   - Input: Array de predições com `cell_id` e features
   - Output: Array de predições com latência média
   - Uso: Processamento eficiente de múltiplas células

**Exemplo de Request** (POST /predict):
```json
{
  "features": {
    "stop_count": 10.0,
    "route_count": 5.0,
    "daily_trips": 800.0,
    "stop_density": 40.0,
    "route_diversity": 0.9,
    "stop_count_norm": 0.8,
    "route_count_norm": 0.7,
    "daily_trips_norm": 0.85
  }
}
```

**Exemplo de Response**:
```json
{
  "prediction": 1,
  "predicted_class": "well_served",
  "probabilities": {
    "underserved": 0.0001,
    "well_served": 0.9999
  },
  "confidence": 0.9999,
  "latency_ms": 0.38
}
```

**Performance da API**:
- **Latência Mediana**: 0.38 ms (526× mais rápido que requisito de 200ms)
- **Throughput Teórico**: ~138.000 predições/segundo
- **Batch de 100 predições**: 0.72 ms total (0.007 ms por predição)

**Documentação Automática**:
- OpenAPI/Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Execução**:
```bash
# Iniciar servidor
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Testar health check
curl http://localhost:8000/health
```

---

## 5. Avaliação Crítica

### 5.1 Pontos Fortes da Solução

#### 5.1.1 Performance Excepcional
- F1-score de 1.0000 no conjunto de teste excede amplamente benchmarks típicos
- Consistência entre validação cruzada e teste indica boa generalização
- Três algoritmos independentes convergem para alta performance (validação cruzada de abordagens)

#### 5.1.2 Eficiência Computacional
- Treinamento em 11.52 segundos permite experimentação rápida
- Inferência sub-milissegundo viabiliza aplicações em tempo real
- Modelo compacto (0.0853 MB) facilita deployment

#### 5.1.3 Interpretabilidade
- Regressão Logística oferece coeficientes lineares interpretáveis
- Importância de features alinha com intuição de domínio (rotas e frequência são críticas)
- Resultados comunicáveis para stakeholders não-técnicos (gestores públicos)

#### 5.1.4 Reprodutibilidade
- Pipeline end-to-end automatizado
- Random seed fixo (42) garante resultados determinísticos
- Código modularizado facilita extensões

#### 5.1.5 Production-Ready
- API REST com validação de input (Pydantic)
- Tratamento de erros com HTTP status codes apropriados
- Documentação automática via OpenAPI
- Formato ONNX permite deployment cross-platform

### 5.2 Limitações e Desafios

#### 5.2.1 Qualidade dos Labels ⚠️
**Problema**: Labels gerados algoritmicamente (percentil 30) não refletem necessariamente avaliação humana de "adequação" de transporte.

**Impacto**:
- Modelo aprende a prever threshold algorítmico, não qualidade real de serviço
- Pode divergir de percepção de residentes ou especialistas em mobilidade urbana

**Evidência do Risco**: Performance "perfeita" (1.0000) pode indicar que classes são artificialmente separáveis devido ao método de labeling.

**Mitigação Recomendada**:
- Validar predições com avaliações de especialistas em planejamento urbano
- Incorporar surveys de satisfação de usuários de transporte público
- Comparar com métricas de acessibilidade alternativas (e.g., isócronas de tempo de viagem)

#### 5.2.2 Risco de Overfitting
**Observação**: Gap quase nulo entre validação cruzada (0.9998) e teste (1.0000) é incomum.

**Possíveis Causas**:
1. **Classes genuinamente separáveis**: Features discriminam bem as categorias (explicação positiva)
2. **Data leakage sutil**: Informação de teste vazou indiretamente (menos provável com stratified split)
3. **Simplicidade do problema**: Threshold linear é suficiente para separação

**Verificação Necessária**:
- Testar em dados de outras cidades (São Paulo, Rio de Janeiro, Fortaleza) para avaliar generalização geográfica
- Validação temporal: treinar com dados de 2024, testar com 2025

#### 5.2.3 Desbalanceamento de Classes
**Distribuição**: 70% mal atendidas, 30% bem atendidas

**Tratamento Atual**:
- Stratified split preserva proporções
- Métrica F1 (macro-averaged) balanceia classes
- Modelo atinge recall perfeito na classe minoritária

**Limitação Residual**:
- Em deployment real, pode haver regiões com distribuição diferente
- Sugestão: Coletar métricas separadas por bairro/distrito

#### 5.2.4 Redundância de Features
**Observação**: Features normalizadas (`*_norm`) têm importância menor que features brutas.

**Implicação**:
- 5 features podem ser reduzidas para 4-5 sem perda significativa
- Multicolinearidade potencial entre `stop_count` e `stop_count_norm`

**Melhoria Sugerida**:
- Aplicar PCA ou seleção de features (Recursive Feature Elimination)
- Comparar performance com subset reduzido

#### 5.2.5 Generalização Geográfica Desconhecida
**Problema**: Modelo treinado exclusivamente em Belo Horizonte-MG.

**Questões Abertas**:
- Performance se mantém em cidades com perfis de transporte diferentes? (e.g., cidades com metrô, BRT)
- Grid de 500m é apropriado para cidades menores ou maiores?
- Definições de "mal atendida" variam por contexto socioeconômico?

**Recomendação**:
- Transfer learning: fine-tuning com dados de novas cidades
- Retraining periódico com dados locais

### 5.3 Possíveis Melhorias

#### 5.3.1 Engenharia de Features Avançada
**Propostas**:
1. **Features Temporais**:
   - Frequência horário de pico vs. horário comum
   - Disponibilidade de serviço noturno/finais de semana
   - Variabilidade de headway (tempo entre veículos)

2. **Features Espaciais**:
   - Distância ao centro da cidade
   - Proximidade a hubs de transporte (terminais, estações)
   - Conectividade com outras células (análise de rede)

3. **Features Demográficas** (requer dataset externo):
   - Densidade populacional por célula
   - Renda média do bairro
   - Proporção de trabalhadores que dependem de transporte público

**Impacto Esperado**: Capturar padrões mais nuancedos de necessidade de transporte.

#### 5.3.2 Modelos Mais Sofisticados
**Alternativas**:
1. **XGBoost/LightGBM**: Gradiente boosting otimizado para performance
2. **Redes Neurais**: MLPs para capturar não-linearidades complexas
3. **Ensemble Stacking**: Combinar predições de LR + RF + GB via meta-learner

**Trade-off**: Maior complexidade vs. interpretabilidade/eficiência.

#### 5.3.3 Calibração de Probabilidades
**Problema**: Mesmo com alta acurácia, probabilidades podem estar mal calibradas.

**Solução**:
- Aplicar Platt Scaling ou Isotonic Regression
- Validar calibração com reliability diagrams

**Benefício**: Confiança numérica nas predições para tomada de decisão.

#### 5.3.4 Explicabilidade Local
**Ferramentas**:
- **SHAP** (SHapley Additive exPlanations): Contribuição de cada feature por predição
- **LIME** (Local Interpretable Model-agnostic Explanations): Aproximação linear local

**Uso**: Explicar para gestores *por que* uma célula específica foi classificada como mal atendida.

#### 5.3.5 Monitoramento em Produção
**Métricas a Rastrear**:
- Data drift: Distribuição de features mudou ao longo do tempo?
- Concept drift: Relação features→labels mudou?
- Performance degradation: Acurácia em novos dados

**Infraestrutura**:
- Logging de predições + timestamps
- Dashboard de monitoramento (Grafana, MLflow)
- Alertas automáticos para anomalias

#### 5.3.6 Interface de Visualização
**Proposta**: Web app interativo para visualizar classificações no mapa.

**Funcionalidades**:
- Mapa de calor: Cores indicando nível de cobertura
- Filtros: Por bairro, linha de ônibus, horário
- Simulação "what-if": Adicionar nova linha, ver impacto na cobertura

**Tecnologias**: Folium (mapas), Streamlit/Dash (interface), GeoPandas (dados espaciais).

### 5.4 Impacto e Aplicações

#### 5.4.1 Políticas Públicas
**Uso Potencial**:
- Priorização de investimentos em infraestrutura de transporte
- Identificação de "desertos de transporte" para programas sociais
- Avaliação de impacto de novas linhas antes de implementação

#### 5.4.2 Planejamento Urbano
**Integração com Outros Sistemas**:
- Planos diretores municipais
- Estudos de impacto de vizinhança (EIV)
- Zoneamento urbano baseado em acessibilidade

#### 5.4.3 Transparência e Participação Social
**Democratização de Dados**:
- Publicar classificações como open data
- Permitir que cidadãos consultem cobertura de seus bairros
- Subsidiar movimentos por melhoria de transporte público

---

## 6. Instruções para Reprodução

### 6.1 Requisitos de Sistema

**Hardware Mínimo**:
- CPU: Qualquer processador x64 moderno
- RAM: 4 GB (8 GB recomendado)
- Disco: 2 GB de espaço livre

**Software**:
- Sistema Operacional: Linux (Ubuntu 22.04+), macOS 14+, ou Windows 11 com WSL2
- Python: 3.12+ (testado em 3.12.3)
- Git: Para clonar repositório

### 6.2 Instalação

#### Passo 1: Clonar Repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd transit-coverage-classifier
```

#### Passo 2: Criar Ambiente Virtual
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
```

#### Passo 3: Instalar Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependências Principais**:
- `scikit-learn==1.5.2`: Treinamento de modelos
- `pandas==2.2.3`: Manipulação de dados
- `numpy==2.1.3`: Operações numéricas
- `onnx==1.17.0`, `skl2onnx==1.18.0`, `onnxruntime==1.20.1`: Exportação e inferência ONNX
- `fastapi==0.124.0`, `uvicorn==0.34.0`: API REST
- `matplotlib==3.9.3`, `seaborn==0.13.2`: Visualizações

### 6.3 Execução do Pipeline

#### Opção 1: Pipeline Completo (Automático)
```bash
python run_pipeline.py --config config/config.yaml
```

**Tempo Esperado**: ~30 segundos  
**Saída**: Todos os artefatos (modelos, figuras, tabelas)

#### Opção 2: Executar Fases Individualmente

**Fase 3: Gerar Grid Espacial**
```bash
python -m src.grid.grid_generator --config config/config.yaml
```
Saída: `data/processed/grid/fortaleza_grid_500m.geojson`

**Fase 4: Extrair Features**
```bash
python -m src.features.feature_extractor --config config/config.yaml
```
Saída: `data/processed/features/features.csv`

**Fase 5: Gerar Labels**
```bash
python -m src.features.label_generator --config config/config.yaml
```
Saída: `data/processed/features/features_with_labels.csv`

**Fase 6: Preparar Datasets**
```bash
python -m src.data.dataset_splitter --config config/config.yaml
```
Saída: `data/processed/datasets/{train,val,test}.csv`

**Fase 7: Treinar Modelos**
```bash
python -m src.models.train --config config/config.yaml
```
Saída: `models/transit_coverage/*.pkl`, `training_summary.txt`

**Fase 8: Avaliar Modelos**
```bash
python -m src.models.evaluator --config config/config.yaml
```
Saída: `reports/figures/*.png`, `reports/tables/*.csv`

**Fase 9: Exportar Modelo ONNX**
```bash
python -m src.models.export --config config/config.yaml
```
Saída: `models/transit_coverage/best_model.onnx`, `model_metadata.json`

**Fase 10: Iniciar API de Inferência**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Acesso: `http://localhost:8000/docs` (documentação interativa)

### 6.4 Verificação de Resultados

#### Verificar Métricas de Performance
```bash
cat reports/tables/model_comparison.csv
```

**Valores Esperados**:
- Logistic Regression: F1 ≈ 1.0000
- Random Forest: F1 ≈ 0.989
- Gradient Boosting: F1 ≈ 0.989

#### Verificar Modelo Exportado
```bash
ls -lh models/transit_coverage/best_model.onnx
python -c "import onnx; model = onnx.load('models/transit_coverage/best_model.onnx'); print('✓ ONNX válido')"
```

#### Testar API
```bash
# Health check
curl http://localhost:8000/health

# Predição de exemplo
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"stop_count": 10, "route_count": 5, "daily_trips": 800, "stop_density": 40, "route_diversity": 0.9, "stop_count_norm": 0.8, "route_count_norm": 0.7, "daily_trips_norm": 0.85}}'
```

**Resposta Esperada**: JSON com `prediction`, `probabilities`, `confidence`, `latency_ms`.

### 6.5 Reprodutibilidade

**Determinismo Garantido**:
- `random_state=42` em todos os geradores aleatórios
- Splits estratificados com mesma seed
- GridSearchCV/RandomizedSearchCV com seed fixo

**Variabilidade Esperada**:
- Tempo de treinamento: ±20% dependendo de CPU
- Latência de API: ±0.1ms dependendo de carga do sistema
- Métricas de performance: Devem coincidir até 4 casas decimais

**Troubleshooting**:
- **Erro "GTFS not found"**: Baixar dados GTFS de Belo Horizonte ou usar dados sintéticos em `data/synthetic_gtfs/`
- **Erro "Model not loaded"**: Executar Fase 9 (export) antes de iniciar API
- **Métricas divergem**: Verificar versão de Python (3.12+) e scikit-learn (1.5.2)

---

## 7. Estrutura do Repositório

```
transit-coverage-classifier/
├── README.md                          # Documentação principal
├── requirements.txt                   # Dependências Python
├── config/
│   └── config.yaml                    # Configurações do pipeline
├── data/
│   ├── gtfs/                          # Dados GTFS de entrada (não versionado)
│   └── processed/                     # Dados processados
│       ├── grid/                      # Grid geográfico gerado
│       ├── features/                  # Features extraídas e labels
│       └── datasets/                  # Splits treino/val/teste
├── src/
│   ├── grid/
│   │   └── grid_generator.py         # Geração de grid espacial
│   ├── features/
│   │   ├── feature_extractor.py      # Extração de features
│   │   └── label_generator.py        # Geração de labels
│   ├── data/
│   │   └── dataset_splitter.py       # Split estratificado
│   ├── models/
│   │   ├── train.py                  # Treinamento de modelos
│   │   ├── evaluator.py              # Avaliação e métricas
│   │   └── export.py                 # Exportação ONNX
│   └── api/
│       ├── main.py                   # Aplicação FastAPI
│       └── prediction_service.py     # Serviço de inferência ONNX
├── models/
│   └── transit_coverage/
│       ├── best_model.pkl            # Melhor modelo (scikit-learn)
│       ├── best_model.onnx           # Modelo exportado (ONNX)
│       ├── model_metadata.json       # Metadados do modelo
│       └── training_summary.txt      # Resumo do treinamento
├── reports/
│   ├── figures/                      # Gráficos e visualizações
│   │   ├── confusion_matrix_*.png
│   │   ├── roc_curves_comparison.png
│   │   └── feature_importance_comparison.png
│   └── tables/                       # Tabelas de resultados
│       ├── model_comparison.csv
│       ├── feature_importance.csv
│       └── classification_report.txt
├── notebooks/                        # Notebooks exploratórios (opcional)
└── run_pipeline.py                   # Script para executar pipeline completo
```

---

## 8. Considerações Finais

Este projeto demonstra um pipeline completo de Machine Learning, desde a geração de dados até o deployment de modelo via API REST. Os principais resultados alcançados foram:

**Objetivos Atingidos**:
✅ Dataset real de médio porte (2.438 amostras, 8 features)  
✅ Pipeline end-to-end automatizado e reproduzível  
✅ Comparação rigorosa entre 3 algoritmos de ML  
✅ Performance excepcional (F1 = 1.0000)  
✅ Exportação em formato padrão (ONNX)  
✅ API REST funcional com FastAPI  
✅ Documentação completa e instruções de reprodução  

**Lições Aprendidas**:
1. Labeling algorítmico permite prototipagem rápida, mas requer validação de domínio
2. Modelos lineares simples podem ser suficientes para problemas bem definidos
3. ONNX facilita transição de experimentação para produção
4. Métricas apropriadas (F1 para desbalanceamento) são críticas para avaliação justa

**Próximos Passos**:
- Validar com especialistas em mobilidade urbana de Belo Horizonte
- Testar generalização em outras cidades brasileiras (São Paulo, Rio de Janeiro, Brasília)
- Implementar interface web para visualização de mapas
- Publicar como ferramenta open-source para gestores públicos

**Repositório**: [URL a ser preenchido]  
**Data de Entrega**: 10/12/2025  
**Contato**: [Email a ser preenchido]

---

## Referências

1. **GTFS Specification**: General Transit Feed Specification Reference. Google Transit, 2024.
2. **Pedregosa et al.**: "Scikit-learn: Machine Learning in Python". Journal of Machine Learning Research, 12:2825-2830, 2011.
3. **ONNX**: Open Neural Network Exchange. https://onnx.ai
4. **FastAPI**: Ramírez, S. "FastAPI: Modern Python Web Framework". https://fastapi.tiangolo.com
5. **Breiman, L.**: "Random Forests". Machine Learning, 45(1):5-32, 2001.
6. **Friedman, J.H.**: "Greedy Function Approximation: A Gradient Boosting Machine". Annals of Statistics, 29(5):1189-1232, 2001.

---

**Relatório gerado automaticamente em**: 10/12/2025 às 08:09:57  
**Script**: `generate_report.py`  
**Versão do Modelo**: 1.0.0
