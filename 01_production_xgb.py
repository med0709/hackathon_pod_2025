"""
===================================================================================
PIPELINE DE PRODUÇÃO - MODELO XGBOOST
===================================================================================

Script para carregar dados tratados e gerar predições com o modelo XGBoost.
Os dados já estão tratados, apenas carrega, faz predição e gera métricas.

FLUXO:
  1. Carrega dados tratados (abt01_test.parquet ou abt01_controle.parquet)
  2. Carrega modelo XGBoost (modelo_credito_xgb.json)
  3. Gera predições com threshold configurável
  4. Calcula métricas completas (Precision, Recall, F1, AUC, GINI, KS, etc)
  5. Gera gráfico de feature importance
  6. Salva resultados e métricas

ALTERAÇÃO: Mude apenas os PARÂMETROS GLOBAIS no início do script
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
import xgboost as xgb

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Diretórios
from paths import *

# Métricas e validação
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix
)

# Avisos
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
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
VERSAO_MODELO = 'XGBoost Otimizado'

# ================================================================================
# FUNÇÕES DE PROCESSAMENTO 

def carregar_dados_entrada(arquivo):
    """Carrega dados tratados para predição"""
    caminho = PREDICTIONS_DIR / arquivo
    print(f'📂 Carregando dados de: {caminho}')
    
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {caminho}')
    
    df = pd.read_parquet(caminho)
    print(f'✓ {len(df)} registros carregados ({df.shape[1]} colunas)')
    return df


def carregar_modelo_xgb():
    """Carrega modelo XGBoost do arquivo JSON"""
    print('\n📦 Carregando modelo XGBoost...')
    
    caminho_modelo = MODELS_DIR / 'modelo_credito_xgb.json'
    
    if not caminho_modelo.exists():
        raise FileNotFoundError(f'Modelo não encontrado: {caminho_modelo}')
    
    modelo = xgb.Booster()
    modelo.load_model(str(caminho_modelo))
    
    print(f'✓ Modelo carregado: {caminho_modelo.name}')
    return modelo


def gerar_predicoes(modelo, df, threshold=THRESHOLD):
    """
    Gera predições usando o modelo XGBoost:
    - Cria DMatrix para predição
    - Gera probabilidades e classes preditas
    """
    print(f'\n🔮 Gerando predições (threshold={threshold})...')
    
    # Separa features e target (se existir)
    if TARGET in df.columns:
        X = df.drop(columns=[TARGET])
        y_true = df[TARGET].values
        tem_target = True
    else:
        X = df
        y_true = None
        tem_target = False
    
    # Cria DMatrix
    dmatrix = xgb.DMatrix(X)
    
    # Probabilidades (classe 1)
    proba = modelo.predict(dmatrix)
    
    # Classes preditas com threshold customizado
    pred = (proba >= threshold).astype(int)
    
    print(f'✓ Predições geradas para {len(pred)} registros')
    print(f'  Proporção Classe 1: {pred.mean():.2%}')
    
    return X, y_true, proba, pred, tem_target


def calcular_metricas(y_true, proba, pred):
    """
    Calcula todas as métricas de avaliação:
    - Precision, Recall, F1-score
    - AUC-ROC, GINI
    - KS (Kolmogorov-Smirnov)
    - Confusion Matrix (TP, TN, FP, FN)
    """
    print('\n📊 Calculando métricas...')
    
    # Métricas básicas
    precision = precision_score(y_true, pred)
    recall = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    
    # AUC e GINI
    try:
        auc = roc_auc_score(y_true, proba)
        gini = 2 * auc - 1
    except ValueError:
        auc = np.nan
        gini = np.nan
    
    # KS (Kolmogorov-Smirnov)
    try:
        fpr, tpr, _ = roc_curve(y_true, proba)
        ks = np.max(tpr - fpr)
    except:
        ks = np.nan
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    
    metricas = {
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'AUC': auc,
        'GINI': gini,
        'KS': ks,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Versao_Modelo': VERSAO_MODELO,
        'Data_Execucao': datetime.now().strftime('%d-%m-%Y'),
        'Threshold': THRESHOLD
    }
    
    print('✓ Métricas calculadas:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  F1-score: {f1:.4f}')
    print(f'  AUC: {auc:.4f}')
    print(f'  GINI: {gini:.4f}')
    print(f'  KS: {ks:.4f}')
    
    return metricas


def salvar_resultados(df_original, proba, pred):
    """Salva predições em arquivo CSV"""
    print('\n💾 Salvando predições...')
    
    # Cria DataFrame com resultados
    df_pred = df_original.copy()
    df_pred['probabilidade_classe_1'] = proba
    df_pred['classe_predita'] = pred
    
    # Salva predições
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_pred = METRICS_DIR / f'predicoes_xgb_{timestamp}.csv'
    df_pred.to_csv(arquivo_pred, index=False)
    print(f'✓ Predições salvas: {arquivo_pred}')
    
    return df_pred


def salvar_metricas(metricas):
    """Salva ou atualiza métricas no arquivo model_metrics_teste.csv"""
    print('\n💾 Salvando métricas...')
    
    df_metrics = pd.DataFrame([metricas])
    
    path_metrics = METRICS_DIR / 'model_metrics_teste.csv'
    
    # Se arquivo existe, atualiza ou adiciona nova linha
    if path_metrics.exists():
        df_hist = pd.read_csv(path_metrics)
        
        # Verifica se já existe uma entrada para esta versão e dataset
        mask = (
            (df_hist['Versao_Modelo'] == VERSAO_MODELO) &
            (df_hist.get('Threshold', THRESHOLD) == THRESHOLD)
        )
        
        if mask.any():
            # Atualiza linha existente
            df_hist.loc[mask] = df_metrics.iloc[0].values
        else:
            # Adiciona nova linha
            df_hist = pd.concat([df_hist, df_metrics], ignore_index=True)
    else:
        df_hist = df_metrics
    
    df_hist.to_csv(path_metrics, index=False)
    print(f'✅ Métricas salvas em: {path_metrics}')


def plotar_feature_importance(modelo, feature_names, top_n=20):
    """
    Gera e salva gráfico de feature importance
    """
    print(f'\n📈 Gerando gráfico de Feature Importance (top {top_n})...')
    
    # Obtém importâncias
    importance_dict = modelo.get_score(importance_type='gain')
    
    if not importance_dict:
        print('⚠️ Modelo não possui feature importance disponível')
        return
    
    # Cria DataFrame com importâncias
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in importance_dict.items()
    ]).sort_values('importance', ascending=False)
    
    # Extrai índices de features ("f0", "f1", etc.)
    importance_df['feature_idx'] = importance_df['feature'].str.extract('(\d+)', expand=False).astype('Int64')
    
    # Mapeia índices para nomes reais
    feature_map = {i: name for i, name in enumerate(feature_names)}
    importance_df['feature_name'] = importance_df['feature_idx'].map(feature_map)
    
    # Remove features não mapeadas
    importance_df = importance_df.dropna(subset=['feature_name'])
    
    if importance_df.empty:
        print('⚠️ Nenhuma feature foi mapeada corretamente')
        return
    
    # Ordena por importância e pega top N
    importance_df = importance_df.sort_values('importance', ascending=True)
    top_features = importance_df.tail(top_n)
    
    # Cria gráfico
    plt.figure(figsize=(10, max(8, top_n * 0.4)))
    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature_name'])
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Salva gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_plot = GRAPHICS_DIR / f'feature_importance_xgb_{timestamp}.png'
    plt.savefig(arquivo_plot, dpi=300, bbox_inches='tight')
    print(f'✓ Gráfico salvo: {arquivo_plot}')
    
    plt.close()
    
    # Também salva a tabela de importâncias
    arquivo_csv = METRICS_DIR / f'feature_importance_xgb_{timestamp}.csv'
    importance_df.to_csv(arquivo_csv, index=False)
    print(f'✓ Tabela de importâncias salva: {arquivo_csv}')


def imprimir_resumo(metricas):
    """Imprime resumo das métricas"""
    print('\n' + '='*70)
    print('📊 RESUMO DAS MÉTRICAS - MODELO XGBOOST')
    print('='*70)
    print(f"Versão do Modelo: {metricas['Versao_Modelo']}")
    print(f"Data/Hora: {metricas['Data_Execucao']}")
    print(f"Threshold: {metricas['Threshold']:.2f}")
    print('-'*70)
    print(f"Precision:  {metricas['Precision']:.6f}")
    print(f"Recall:     {metricas['Recall']:.6f}")
    print(f"F1-score:   {metricas['F1_score']:.6f}")
    print(f"AUC:        {metricas['AUC']:.6f}")
    print(f"GINI:       {metricas['GINI']:.6f}")
    print(f"KS:         {metricas['KS']:.6f}")
    print('-'*70)
    print(f"True Positives (TP):  {metricas['TP']:,}")
    print(f"True Negatives (TN):  {metricas['TN']:,}")
    print(f"False Positives (FP): {metricas['FP']:,}")
    print(f"False Negatives (FN): {metricas['FN']:,}")
    print('='*70)


# ================================================================================
# EXECUÇÃO PRINCIPAL

def main():
    """Executa pipeline completo"""
    print('\n' + '='*70)
    print('🚀 PIPELINE DE PRODUÇÃO - MODELO XGBOOST - INÍCIO')
    print('='*70)
    
    try:
        # 1. Carrega dados
        df = carregar_dados_entrada(ARQUIVO_ENTRADA)
        
        # 2. Carrega modelo
        modelo = carregar_modelo_xgb()
        
        # 3. Gera predições
        X, y_true, proba, pred, tem_target = gerar_predicoes(modelo, df, THRESHOLD)
        
        # 4. Se tem target, calcula métricas e gera gráficos
        if tem_target:
            metricas = calcular_metricas(y_true, proba, pred)
            imprimir_resumo(metricas)
            salvar_metricas(metricas)
            plotar_feature_importance(modelo, X.columns.tolist())
        else:
            print('\nℹ️ Dataset sem coluna alvo. Apenas predições foram geradas.')
        
        # 5. Salva predições
        df_pred = salvar_resultados(df, proba, pred)
        
        print('\n✅ PIPELINE CONCLUÍDO COM SUCESSO\n')
        return df_pred
        
    except Exception as e:
        print(f'\n❌ ERRO: {str(e)}')
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    resultado = main()
