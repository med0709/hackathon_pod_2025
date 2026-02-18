# Projeto de Predicao - Hackathon Pod

Este projeto contem o fluxo completo de tratamento de dados, selecao de variaveis, treinamento do modelo e predicao em producao.

## Estrutura principal

- configs/ : funcoes e caminhos do projeto
- data/ : dados brutos, processados e predicoes
- artifact/ : artefatos do processamento (features, stats de nulos)
- models/ : modelo salvo (pipeline)
- reports/ : metricas historicas
- notebooks/ : notebooks de desenvolvimento e analise
- production.py : script de producao para gerar predicoes

## Dependencias

Instale as bibliotecas do projeto:

```bash
pip install -r requirements.txt
```

## Fluxo de treinamento (referencia)

1) Tratamento dos dados
- Notebook: notebooks/02_tratamento_dados.ipynb
- Gera:
  - data/processed/abt01_train.csv
  - data/processed/abt00_test.csv
  - artifact/features_stage_01.pkl
  - artifact/stats_nulo.pkl

2) Selecao de variaveis
- Notebook: notebooks/03_Feature_selection.ipynb
- Gera:
  - artifact/features_stage_02.pkl

3) Modelo
- Notebook: notebooks/06_regressao_logistica_sklearn_sem_premissas.ipynb
- Gera:
  - models/logreg_pipeline.pkl
  - reports/model_metrics.csv

## Predicao em producao

Use o script production.py. Ele:
- Le os dados de entrada
- Aplica o mesmo tratamento
- Usa as features do artifact/features_stage_02.pkl
- Aplica o modelo salvo
- Salva predicoes e metricas

### Como executar

1) Ajuste o arquivo de entrada no production.py:

```python
DATA_ENTRADA = PROCESSED_DIR / 'abt00_test.csv'
```

2) Execute o script:

```bash
python production.py
```

### Saidas geradas

- data/predictions/predicoes_YYYYMMDD_HHMMSS.csv
- reports/stats_predicoes_YYYYMMDD_HHMMSS.csv
- reports/model_metrics.csv (se existir a coluna FPD no arquivo de entrada)

## Analise das predicoes

Notebook para visualizar metricas e graficos:
- notebooks/05_analise_predicoes.ipynb

Este notebook:
- Le a ultima predicao em data/predictions
- Mostra metricas (accuracy, precision, recall, f1, roc_auc, pr_auc, gini, ks)
- Mostra matriz de confusao
- Mostra distribuicao de classes e probabilidades
- Mostra event rate por decil (0 e 1)

## Observacoes importantes

- Para calcular metricas de teste, o arquivo de predicao precisa conter a coluna FPD.
- Os artefatos na pasta artifact devem existir antes de rodar o production.py.
- O modelo salvo deve existir em models/logreg_pipeline.pkl.

## Suporte

Se precisar ajustar o fluxo, revise os notebooks de desenvolvimento e atualize os artefatos antes de rodar em producao.
