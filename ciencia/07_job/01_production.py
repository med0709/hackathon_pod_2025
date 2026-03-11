"""
===================================================================================
PIPELINE DE PRODUÇÃO - ANCORAGEM E PREDIÇÃO
===================================================================================

Script para carregar dados âncora, aplicar processamento e gerar predições.
Os dados são processados seguindo o mesmo fluxo dos notebooks de desenvolvimento.

FLUXO:
  1. Carrega dados âncora (abt00_test.parquet)
  2. Aplica transformações do notebook de tratamento 
  3. Seleciona features do arquivo features_stage_02.pkl
  4. Carrega e aplica pipeline do modelo
  5. Gera predições e salva resultados

ALTERAÇÃO: Mude apenas as variáveis
"""

# ================================================================================
# Bibliotecas / Configuração inicial

# Acesso aos modulos do diretório
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# Manipulação dos dados
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import joblib
# Diretórios
from config.paths import *
# Métricas e validação
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
# Avisos
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# Modelos
MAPA_MODELOS = {
    1: "pycaret_model.pkl",
}
# ================================================================================
# Parâmetro Global
TARGET = 'FPD'
THRESHOLD = 0.41
MODELO = {"versao": 1}
VERSAO = MAPA_MODELOS[MODELO["versao"]]

# ================================================================================
# Carregamento dos dados 
# 🔴 ALTERE APENAS AQUI O ARQUIVO DE ENTRADA DE DADOS
# Pode ser um arquivo CSV ou uma lista de linhas para processar
DATA_ENTRADA = PREDICTIONS_DIR / 'base_test.parquet'

# ================================================================================
# FUNÇÕES DE PROCESSAMENTO 

def carregar_dados_entrada(caminho_dados):
    """Carrega dados de entrada para predição"""
    print(f'📂 Carregando dados de: {caminho_dados}')
    
    if not caminho_dados.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {caminho_dados}')
    
    df = pd.read_parquet(caminho_dados)
    print(f'✓ {len(df)} registros carregados ({df.shape[1]} colunas)')
    return df

def carregar_modelos(models_dir: Path, mapa: dict[int, str]) -> dict[int, object]:
    modelos = {}
    for versao, nome_arquivo in mapa.items():
        path = models_dir / nome_arquivo
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
        modelos[versao] = joblib.load(path)
    return modelos

MODELOS = carregar_modelos(MODELS_DIR, MAPA_MODELOS)

def carregar_artefatos():
    """Carrega todos os artefatos necessários para produção"""
    print('\n📦 Carregando artefatos...')
    
    # Features selecionadas
    with open(ARTIFACT_DIR / 'features_stage_02.pkl', 'rb') as f:
        features = pickle.load(f)
    print(f'✓ {len(features)} features carregadas')
    
    # Estatísticas de nulos
    with open(ARTIFACT_DIR / 'stats_nulo.pkl', 'rb') as f:
        stats = pickle.load(f)
    print('✓ Estatísticas de nulos carregadas')
    
    # Pipeline do modelo
    pipe = MODELOS[MODELO["versao"]]
    print(f"✓ Pipeline carregado (versao {MODELO['versao']})")
    
    return features, stats, pipe


# prepara dados para inferencia
def preparar_dados(df, features, stats):
    """
    Aplica transformações de tratamento para produção:
    - Seleciona apenas as features do treino
    - Substitui variações de "Desconhecido" por NaN em colunas categóricas
    - Aplica cap de segurança em SCORE_RATEO [0, 10]
    - Converte colunas não numéricas para Int64 (aceita NaN)
    - Preenche valores nulos com estatísticas do treino
    - Retorna dataset pronto para inferência
    """
    print('\n🔄 Processando dados...')

    df_proc = df.copy()

    # Seleciona apenas as features usadas no treino
    df_proc = df_proc[features].copy()

    # Trata "Desconhecido" (qualquer variação) apenas em colunas categóricas
    cols_obj = df_proc.select_dtypes(include='object').columns

    cols_com_desconhecido = [
        col for col in cols_obj
        if df_proc[col].astype(str).str.strip().str.match(r'(?i)^desconhecido$').any()
    ]

    for col in cols_com_desconhecido:
        df_proc[col] = (
            df_proc[col]
            .astype(str)
            .str.strip()
            .replace(r'(?i)^desconhecido$', np.nan, regex=True)
        )

    # Saneia SCORE_RATEO com cap [0, 10]
    if 'SCORE_RATEO' in df_proc.columns:
        df_proc['SCORE_RATEO'] = df_proc['SCORE_RATEO'].clip(0, 10)

    # Converte colunas não numéricas para inteiro que aceita NaN
    cols_nao_numericas = df_proc.select_dtypes(exclude=['float', 'int']).columns.tolist()
    for col in cols_nao_numericas:
        df_proc[col] = df_proc[col].astype('Int64')

    # Preenche nulos com estatísticas do treino
    for col, valor in stats['numerical'].items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(valor)

    for col in stats['categorical_cols']:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(stats['categorical_fill'])

    print(f'✓ Dados preparados: {df_proc.shape[0]} registros x {df_proc.shape[1]} features')
    return df_proc


def gerar_predicoes(X, pipe, threshold=THRESHOLD):
    """
    Gera predições usando o pipeline (scaler + modelo):
    - Normaliza os dados
    - Aplica o modelo
    - Gera probabilidades e classes preditas
    """
    print('\n🔮 Gerando predições...')
    
    # Probabilidades (classe 1)
    proba = pipe.predict_proba(X)[:, 1]
    
    # Classes preditas com threshold customizado
    pred = (proba >= threshold).astype(int)
    
    print(f'✓ Predições geradas para {len(pred)} registros')
    print(f'  Proporção Classe 1: {pred.mean():.2%}')
    
    return proba, pred


def salvar_resultados(df_original, X, proba, pred):
    """Salva predições e gera relatório de métricas"""
    print('\n💾 Salvando resultados...')
    
    # Cria DataFrame com resultados
    df_pred = df_original.copy()
    df_pred['probabilidade_classe_1'] = proba
    df_pred['classe_predita'] = pred
    
    # Salva predições
    arquivo_pred = PREDICTIONS_DIR / f'predicoes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_pred.to_csv(arquivo_pred, index=False)
    print(f'✓ Predições salvas: {arquivo_pred}')
    
    # Gera estatísticas
    stats_pred = {
        'data_execucao': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'total_registros': len(pred),
        'classe_0': int((pred == 0).sum()),
        'classe_1': int((pred == 1).sum()),
        'proporcao_classe_1': float(pred.mean()),
        'prob_media': float(proba.mean()),
        'prob_min': float(proba.min()),
        'prob_max': float(proba.max()),
    }
    
    # Salva estatísticas em CSV
    arquivo_stats = METRICS_DIR / f'stats_predicoes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    pd.DataFrame([stats_pred]).to_csv(arquivo_stats, index=False)
    print(f'✓ Estatísticas salvas: {arquivo_stats}')
    
    return df_pred, stats_pred


def atualizar_metricas_modelo(df_original, proba, pred):
    """Atualiza o model_metrics.csv com métricas de teste, se houver TARGET."""
    if TARGET not in df_original.columns:
        print('ℹ️ Coluna alvo não encontrada. Métricas de teste não foram geradas.')
        return None

    y_true = df_original[TARGET].astype(int).values

    precision = precision_score(y_true, pred)
    recall = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)

    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    metrics_row = {
        'Dataset': 'Teste',
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'AUC': auc,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Versao_Modelo': VERSAO,
        'Data_Execucao': datetime.now().strftime('%d-%m-%Y'),
    }

    df_metrics_test = pd.DataFrame([metrics_row])

    path_metrics = METRICS_DIR / 'model_metrics_teste.csv'
    if path_metrics.exists():
        df_hist = pd.read_csv(path_metrics)

        mask = (
            (df_hist['Versao_Modelo'] == VERSAO) &
            (df_hist['Dataset'] == df_metrics_test.iloc[0]['Dataset'])
        )

        if mask.any():
            df_hist.loc[mask] = df_metrics_test.iloc[0].values
        else:
            df_hist = pd.concat([df_hist, df_metrics_test], ignore_index=True)
    else:
        df_hist = df_metrics_test

    df_hist.to_csv(path_metrics, index=False)
    print(f'✅ Métricas (Teste) salvas em: {path_metrics}')
    return df_metrics_test


def imprimir_resumo(stats_pred):
    """Imprime resumo das predições"""
    print('\n' + '='*50)
    print('📊 RESUMO DAS PREDIÇÕES')
    print('='*50)
    print(f"Data/Hora: {stats_pred['data_execucao']}")
    print(f"Total de registros: {stats_pred['total_registros']:,}")
    print(f"Classe 0 (Bom): {stats_pred['classe_0']:,} ({(stats_pred['classe_0']/stats_pred['total_registros']*100):.1f}%)")
    print(f"Classe 1 (Mau): {stats_pred['classe_1']:,} ({(stats_pred['classe_1']/stats_pred['total_registros']*100):.1f}%)")
    print(f"Probabilidade média: {stats_pred['prob_media']:.4f}")
    print(f"Probabilidade min-max: [{stats_pred['prob_min']:.4f}, {stats_pred['prob_max']:.4f}]")
    print('='*50)


# ==================== EXECUÇÃO PRINCIPAL ====================

def main():
    """Executa pipeline completo"""
    print('\n' + '='*50)
    print('🚀 PIPELINE DE PRODUÇÃO - INÍCIO')
    print('='*50)
    
    try:
        # 1. Carrega dados
        df = carregar_dados_entrada(DATA_ENTRADA)
        
        # 2. Carrega artefatos
        features, stats, pipe = carregar_artefatos()
        
        # 3. Processa dados
        X = preparar_dados(df, features, stats)
        
        # 4. Gera predições
        proba, pred = gerar_predicoes(X, pipe)
        
        # 5. Salva resultados
        df_pred, stats_pred = salvar_resultados(df, X, proba, pred)

        # 6. Salva métricas do teste (se houver TARGET)
        atualizar_metricas_modelo(df, proba, pred)
        
        # 7. Resumo
        imprimir_resumo(stats_pred)
        
        print('\n✅ PIPELINE CONCLUÍDO COM SUCESSO\n')
        return df_pred
        
    except Exception as e:
        print(f'\n❌ ERRO: {str(e)}')
        raise


if __name__ == '__main__':
    resultado = main()
