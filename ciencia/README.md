# Projeto de Classificação - Migração Pré-Pago para Pós-Pago

## 📋 Descrição do Projeto

Projeto de classificação para identificar clientes de planos pré-pagos que podem migrar com segurança para planos pós-pagos, minimizando o risco de inadimplência.

## 🎯 Objetivo

Desenvolver um modelo de classificação que identifique clientes com baixo risco de se tornarem inadimplentes (mau pagadores) após migrarem de plano pré-pago para pós-pago.

## 📊 Contexto do Problema

- **Target:** FPD (First Payment Default) - 0 = Bom pagador, 1 = Inadimplente
- **Total de registros:** 1.272.095 clientes
- **Variáveis:** 108 features
- **Taxa de inadimplência:** ~23.4%
- **Custo por linha:** R$ 50,00
- **Período dos dados:** Out/2024 a Mar/2025 (SAFRAs 202410 a 202503)
- **Divisão temporal:** Teste em Fev e Mar/2025, Treino no restante

## 📁 Estrutura do Projeto

```
tel/
├── configs/
│   ├── paths.py                    # Configuração de caminhos
│   ├── function_basic.py           # Funções básicas de análise
│   └── function_others.py          # Funções auxiliares
├── data/
│   ├── raw/
│   │   └── base_tabelao.parquet   # Dados brutos
│   ├── processed/                  # Dados processados
│   └── predictions/                # Predições dos modelos
├── models/                         # Modelos treinados (.pkl)
├── notebooks/
│   ├── 01_entendimento_dados.ipynb
│   ├── 02_tratamento_dados.ipynb
│   ├── 03_regra_negocio.ipynb
│   ├── 04_baseline_pycaret.ipynb
│   ├── 05_modelagem_gridsearch.ipynb
│   └── 06_apresentacao_resultados.ipynb
├── reports/
│   └── metrics/                    # Métricas dos modelos (JSON)
├── artifact/                       # Artefatos (gráficos, tabelas, etc.)
├── requirements.txt
└── README.md
```

## 🔧 Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

## 📚 Notebooks

### 1. Entendimento dos Dados
**Arquivo:** `01_entendimento_dados.ipynb`

- Análise exploratória dos dados
- Distribuição da variável target
- Análise de valores faltantes
- Correlações e estatísticas descritivas

**Principais Insights:**
- Dataset com 1.272.095 registros
- 108 variáveis (numéricas e categóricas)
- Desbalanceamento moderado (76.6% bons vs 23.4% maus)

### 2. Tratamento dos Dados
**Arquivo:** `02_tratamento_dados.ipynb`

- Remoção de colunas com alto percentual de missing
- Tratamento de valores faltantes (mediana para numéricas)
- Feature engineering (scores médios, flags combinadas)
- Encoding de variáveis categóricas
- **Divisão treino/teste TEMPORAL por SAFRA:**
  - **Teste:** Fevereiro e Março de 2025 (SAFRA 202502 e 202503) - ~31% dos dados
  - **Treino:** Outubro, Novembro, Dezembro/2024 e Janeiro/2025 - ~69% dos dados

**Resultados:**
- Dados limpos e prontos para modelagem
- Features salvas em formato Parquet
- Encoders salvos para uso em produção
- **Divisão temporal garante teste em dados futuros (evita data leakage)**

### 3. Regra de Negócio
**Arquivo:** `03_regra_negocio.ipynb`

Implementação e teste de regras de negócio simples:

- **Regra 1:** Score mínimo (SCORE_01 >= threshold)
- **Regra 2:** Score médio
- **Regra 3:** Combinação de múltiplas regras

**Análise de trade-offs:**
- Taxa de aprovação vs risco
- Impacto financeiro por threshold

### 4. Baseline com PyCaret
**Arquivo:** `04_baseline_pycaret.ipynb`

- Comparação automática de múltiplos modelos
- Seleção dos top 3 modelos
- Tuning automático do melhor modelo
- Avaliação completa (métricas, matriz de confusão, ROC)
- Feature importance

**Modelos testados:**
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- E outros...

### 5. Modelagem com Grid Search
**Arquivo:** `05_modelagem_gridsearch.ipynb`

Otimização dos melhores modelos usando Grid Search:

- Definição de grids de hiperparâmetros
- Grid Search com validação cruzada
- Comparação com baseline
- Seleção do melhor modelo final

**Modelos otimizados:**
- Logistic Regression
- Random Forest
- XGBoost

### 6. Apresentação de Resultados
**Arquivo:** `06_apresentacao_resultados.ipynb`

Consolidação e apresentação de todos os resultados:

- Comparação de todos os modelos
- Análise financeira detalhada
- Trade-off aprovação vs risco
- Recomendação final
- Insights e conclusões

## 📊 Métricas de Avaliação

Para cada modelo, avaliamos:

- **Accuracy:** Proporção de acertos geral
- **Precision:** Dos aprovados, quantos são bons pagadores
- **Recall:** Dos maus pagadores, quantos identificamos
- **F1-Score:** Média harmônica de precision e recall
- **ROC-AUC:** Área sob a curva ROC
- **Resultado Líquido:** Impacto financeiro em R$

## 💰 Análise Financeira

**Premissas:**
- Custo por linha telefônica: R$ 50,00
- Ganho: Aprovar clientes bons pagadores
- Perda: Aprovar clientes inadimplentes
- Perda de oportunidade: Rejeitar clientes bons

**Matriz de Confusão Financeira:**
- **Verdadeiro Negativo (TN):** Aprovamos bom pagador → Ganho R$ 50,00
- **Falso Positivo (FP):** Aprovamos mau pagador → Perda R$ 50,00
- **Verdadeiro Positivo (TP):** Rejeitamos mau pagador → Não perdemos
- **Falso Negativo (FN):** Rejeitamos bom pagador → Perda de oportunidade R$ 50,00

## 🏆 Resultados

Os resultados específicos estão disponíveis nos notebooks e nos arquivos:
- `reports/metrics/` - Métricas em JSON
- `artifact/` - Gráficos e tabelas comparativas
- `data/predictions/` - Predições de cada modelo

## 🚀 Como Executar

### Ordem de execução dos notebooks:

1. **Entendimento dos Dados**
   ```bash
   jupyter notebook notebooks/01_entendimento_dados.ipynb
   ```

2. **Tratamento dos Dados**
   ```bash
   jupyter notebook notebooks/02_tratamento_dados.ipynb
   ```

3. **Regra de Negócio**
   ```bash
   jupyter notebook notebooks/03_regra_negocio.ipynb
   ```

4. **Baseline PyCaret**
   ```bash
   jupyter notebook notebooks/04_baseline_pycaret.ipynb
   ```

5. **Modelagem Grid Search**
   ```bash
   jupyter notebook notebooks/05_modelagem_gridsearch.ipynb
   ```

6. **Apresentação de Resultados**
   ```bash
   jupyter notebook notebooks/06_apresentacao_resultados.ipynb
   ```

## 📌 Recomendações

### Para Produção:
1. **Monitoramento contínuo** da performance do modelo
2. **Ajuste de threshold** conforme apetite ao risco do negócio
3. **Retreinamento periódico** (sugestão: trimestral)
4. **A/B testing** para validar impacto real
5. **Coleta de feedback** dos clientes aprovados/rejeitados

### Melhorias Futuras:
- Incluir Customer Lifetime Value (CLV) na análise financeira
- Testar modelos ensemble mais complexos
- Análise de drift dos dados
- Implementar explicabilidade com SHAP/LIME
- Considerar fatores sazonais

## ⚠️ Limitações

- Dataset desbalanceado (requer técnicas de balanceamento)
- Variáveis anonimizadas limitam interpretação
- Período de dados limitado (6 meses: Out/2024 a Mar/2025)

## ✨ Diferenciais do Projeto

- **Divisão temporal por SAFRA:** Teste em meses futuros para simular produção real
- **Análise de regra de negócio vs ML:** Comparação com abordagem tradicional
- **Análise financeira completa:** Impacto em R$ para tomada de decisão
- **Pipeline completo:** Da análise exploratória até recomendação final
- Não considera valor temporal do dinheiro
- Dados históricos podem não refletir comportamento futuro

## 👥 Informações de Contato

Para dúvidas sobre o projeto, consulte a documentação nos notebooks ou o fichamento de dúvidas com o cliente.

## 📝 Licença

Este projeto foi desenvolvido para análise de migração de planos de telecomunicações.

---

**Última atualização:** Janeiro 2026

**Status do Projeto:** ✅ Completo
