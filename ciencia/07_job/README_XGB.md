# 📊 Pipeline de Produção - Modelo XGBoost

Scripts para geração de predições e análises com o modelo XGBoost otimizado.

## 📁 Arquivos Criados

### 1. `02_production_xgb.py`
**Pipeline completo de predição**

Executa todo o fluxo de predição com o modelo XGBoost:
- ✅ Carrega dados tratados (sem necessidade de processamento adicional)
- ✅ Carrega modelo XGBoost (`modelo_credito_xgb.json`)
- ✅ Gera predições com threshold configurável
- ✅ Calcula métricas completas (Precision, Recall, F1, AUC, GINI, KS, TP, TN, FP, FN)
- ✅ Gera gráfico de Feature Importance
- ✅ Salva resultados e métricas

### 2. `03_graficos_xgb.py`
**Geração de gráficos e análises visuais**

Gera visualizações detalhadas do modelo:
- 📈 Feature Importance (Top 20)
- 📈 Curva ROC com AUC
- 📈 Distribuição de Scores por Classe
- 📈 Confusion Matrix
- 📈 KS Plot (Kolmogorov-Smirnov)
- 📈 Precision-Recall Curve

---

## ⚙️ Como Usar

### 🔴 Configuração dos Parâmetros

Ambos os scripts possuem uma seção **PARÂMETROS GLOBAIS** no início que você deve configurar:

```python
# ⚙️ PARÂMETROS GLOBAIS - CONFIGURE AQUI
# ================================================================================

# 🔴 Nome do arquivo de entrada (na pasta PREDICTIONS_DIR)
# Opções: 'abt01_test.parquet' ou 'abt01_controle.parquet'
ARQUIVO_ENTRADA = 'abt01_test.parquet'

# 🔴 Threshold para classificação (ajuste conforme necessário)
THRESHOLD = 0.41

# Nome da coluna alvo
TARGET = 'FPD'

# Versão do modelo
VERSAO_MODELO = 'V2 - XGBoost Otimizado'
```

### 📊 Executar Pipeline de Predição

```bash
# No terminal, navegue até a pasta 07_job
cd c:\Users\billy.reis\Desktop\hackathon_dev\07_job

# Execute o script
python 02_production_xgb.py
```

**Saída:**
- Arquivo CSV com predições: `predictions/predicoes_xgb_YYYYMMDD_HHMMSS.csv`
- Métricas atualizadas: `06_reports/model_metrics_teste.csv`
- Gráfico de Feature Importance: `06_reports/feature_importance_xgb_YYYYMMDD_HHMMSS.png`
- Tabela de importâncias: `06_reports/feature_importance_xgb_YYYYMMDD_HHMMSS.csv`

### 🎨 Executar Geração de Gráficos

```bash
# Execute o script de gráficos
python 03_graficos_xgb.py
```

**Saída:**
Todos os gráficos salvos em `06_reports/`:
- `feature_importance_YYYYMMDD_HHMMSS.png`
- `roc_curve_YYYYMMDD_HHMMSS.png`
- `score_distribution_YYYYMMDD_HHMMSS.png`
- `confusion_matrix_YYYYMMDD_HHMMSS.png`
- `ks_plot_YYYYMMDD_HHMMSS.png`
- `precision_recall_curve_YYYYMMDD_HHMMSS.png`

---

## 📋 Estrutura dos Dados de Entrada

Os dados devem estar na pasta `01_data/predictions/`:
- ✅ **`abt01_test.parquet`** - Base de teste
- ✅ **`abt01_controle.parquet`** - Base de controle

**Requisitos:**
- ✅ Dados já tratados (não requer processamento adicional)
- ✅ Devem conter a coluna `FPD` (target) para calcular métricas
- ✅ Se não houver target, apenas predições serão geradas

---

## 📊 Métricas Geradas

O pipeline calcula as seguintes métricas:

| Métrica | Descrição |
|---------|-----------|
| **Precision** | Proporção de predições positivas corretas |
| **Recall** | Proporção de casos positivos identificados |
| **F1-score** | Média harmônica entre Precision e Recall |
| **AUC** | Área sob a curva ROC |
| **GINI** | 2 * AUC - 1 |
| **KS** | Estatística Kolmogorov-Smirnov |
| **TP, TN, FP, FN** | Valores da matriz de confusão |

---

## 🔧 Ajustando o Threshold

O threshold determina o ponto de corte para classificação:

```python
THRESHOLD = 0.41  # Ajuste conforme necessário
```

- **Threshold mais alto** → Menos predições positivas (maior precisão, menor recall)
- **Threshold mais baixo** → Mais predições positivas (menor precisão, maior recall)

Você pode testar diferentes thresholds e comparar as métricas geradas.

---

## 📝 Formato da Saída de Predições

O arquivo CSV gerado contém:

```csv
<todas as colunas originais>, probabilidade_classe_1, classe_predita
```

Exemplo:
```
feature1, feature2, ..., FPD, probabilidade_classe_1, classe_predita
0.5, 1.2, ..., 1, 0.7234, 1
0.3, 0.8, ..., 0, 0.2156, 0
```

---

## 📝 Formato das Métricas

O arquivo `model_metrics_teste.csv` é atualizado com:

```csv
Precision, Recall, F1_score, AUC, GINI, KS, TP, TN, FP, FN, Versao_Modelo, Data_Execucao, Threshold
0.589163, 0.07493, 0.132951, 0.712737, 0.425474, 0.308351, 6611, 296711, 4610, 81618, V2 - XGBoost Otimizado, 05-03-2026, 0.41
```

---

## 🚀 Fluxo Completo Recomendado

1. **Configure os parâmetros** nos scripts
2. **Execute o pipeline de predição**: `python 02_production_xgb.py`
3. **Analise os resultados** no terminal e no arquivo de métricas
4. **Gere os gráficos**: `python 03_graficos_xgb.py`
5. **Ajuste o threshold** se necessário e repita

---

## ⚠️ Observações Importantes

- ✅ Os dados de entrada **devem estar tratados** (os scripts não aplicam tratamento)
- ✅ O modelo XGBoost deve estar em `05_models/modelo_credito_xgb.json`
- ✅ Se a coluna `FPD` não existir, apenas predições serão geradas (sem métricas)
- ✅ Os arquivos de saída têm timestamp para evitar sobrescrita

---

## 📞 Estrutura de Diretórios

```
hackathon_dev/
├── 01_data/
│   └── predictions/
│       ├── abt01_test.parquet       ← Dados de entrada
│       └── abt01_controle.parquet   ← Dados de entrada
├── 05_models/
│   └── modelo_credito_xgb.json      ← Modelo XGBoost
├── 06_reports/
│   ├── model_metrics_teste.csv      ← Métricas (saída)
│   └── *.png                        ← Gráficos (saída)
└── 07_job/
    ├── 02_production_xgb.py         ← Pipeline de predição
    └── 03_graficos_xgb.py           ← Geração de gráficos
```

---

## ✨ Melhorias Futuras

- [ ] Adicionar análise de deciles
- [ ] Gerar relatório PDF automatizado
- [ ] Adicionar análise de lift e gains
- [ ] Implementar monitoramento de drift

---

**Criado em:** 05/03/2026  
**Versão:** 1.0
