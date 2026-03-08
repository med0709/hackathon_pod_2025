"""
===================================================================================
GERAÇÃO DE GRÁFICOS - MODELO XGBOOST
===================================================================================

Script para gerar visualizações e análises gráficas do modelo XGBoost:
  - Feature Importance
  - Curva ROC
  - Distribuição de Probabilidades
  - Confusion Matrix
  - KS Plot
  - Precision-Recall Curve

ALTERAÇÃO: Configure os PARÂMETROS GLOBAIS no início
"""

# ================================================================================
# Bibliotecas / Configuração inicial

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
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

# Métricas
from sklearn.metrics import (
    roc_curve, 
    auc, 
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

# Avisos
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ================================================================================
# ⚙️ PARÂMETROS GLOBAIS - CONFIGURE AQUI
# ================================================================================

# 🔴 Nome do arquivo de entrada (na pasta PREDICTIONS_DIR)
ARQUIVO_ENTRADA = 'abt01_test.parquet'

# Nome da coluna alvo
TARGET = 'FPD'

# Threshold usado nas predições
THRESHOLD = 0.41

# ================================================================================
# FUNÇÕES DE VISUALIZAÇÃO

def carregar_dados_e_modelo():
    """Carrega dados e modelo para análise"""
    print('📂 Carregando dados e modelo...')
    
    # Carrega dados
    caminho_dados = PREDICTIONS_DIR / ARQUIVO_ENTRADA
    df = pd.read_parquet(caminho_dados)
    
    # Separa X e y
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values
    
    # Carrega modelo
    caminho_modelo = MODELS_DIR / 'modelo_credito_xgb.json'
    modelo = xgb.Booster()
    modelo.load_model(str(caminho_modelo))
    
    # Predições
    dmatrix = xgb.DMatrix(X)
    proba = modelo.predict(dmatrix)
    pred = (proba >= THRESHOLD).astype(int)
    
    print(f'✓ Dados carregados: {len(df)} registros')
    print(f'✓ Modelo carregado: {caminho_modelo.name}')
    
    return X, y, proba, pred, modelo


def plotar_feature_importance(modelo, feature_names, top_n=20):
    """
    Gera gráfico de feature importance
    """
    print(f'\n📈 Gerando Feature Importance (top {top_n})...')
    
    # Obtém importâncias
    importance_dict = modelo.get_score(importance_type='gain')
    
    if not importance_dict:
        print('⚠️ Modelo não possui feature importance disponível')
        return None
    
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
        return None
    
    # Ordena por importância e pega top N
    importance_df = importance_df.sort_values('importance', ascending=True)
    top_features = importance_df.tail(top_n)
    
    # Cria gráfico
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    bars = ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature_name'], fontsize=10)
    ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance - XGBoost Model', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Adiciona valores nas barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo = GRAPHICS_DIR / f'feature_importance_{timestamp}.png'
    plt.savefig(arquivo, bbox_inches='tight')
    print(f'✓ Salvo: {arquivo}')
    plt.close()
    
    return importance_df


def plotar_curva_roc(y_true, proba):
    """
    Gera curva ROC
    """
    print('\n📈 Gerando Curva ROC...')
    
    # Calcula ROC
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    
    # Cria gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo = GRAPHICS_DIR / f'roc_curve_{timestamp}.png'
    plt.savefig(arquivo, bbox_inches='tight')
    print(f'✓ Salvo: {arquivo}')
    plt.close()


def plotar_distribuicao_scores(y_true, proba, threshold=THRESHOLD):
    """
    Gera histograma da distribuição de scores por classe
    """
    print('\n📈 Gerando Distribuição de Scores...')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scores para cada classe
    scores_classe_0 = proba[y_true == 0]
    scores_classe_1 = proba[y_true == 1]
    
    # Histogramas
    ax.hist(scores_classe_0, bins=50, alpha=0.6, color='green', label='Classe 0 (Bom)', edgecolor='black')
    ax.hist(scores_classe_1, bins=50, alpha=0.6, color='red', label='Classe 1 (Mau)', edgecolor='black')
    
    # Linha do threshold
    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
    
    ax.set_xlabel('Probabilidade Predita (Classe 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição de Scores de Probabilidade por Classe', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo = GRAPHICS_DIR / f'score_distribution_{timestamp}.png'
    plt.savefig(arquivo, bbox_inches='tight')
    print(f'✓ Salvo: {arquivo}')
    plt.close()


def plotar_confusion_matrix(y_true, pred):
    """
    Gera matriz de confusão
    """
    print('\n📈 Gerando Confusion Matrix...')
    
    # Calcula confusion matrix
    cm = confusion_matrix(y_true, pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(['Classe 0 (Bom)', 'Classe 1 (Mau)'], fontsize=11)
    ax.set_yticklabels(['Classe 0 (Bom)', 'Classe 1 (Mau)'], fontsize=11)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo = GRAPHICS_DIR / f'confusion_matrix_{timestamp}.png'
    plt.savefig(arquivo, bbox_inches='tight')
    print(f'✓ Salvo: {arquivo}')
    plt.close()


def plotar_ks_plot(y_true, proba):
    """
    Gera gráfico KS (Kolmogorov-Smirnov)
    """
    print('\n📈 Gerando KS Plot...')
    
    # Cria DataFrame
    df = pd.DataFrame({
        'proba': proba,
        'target': y_true
    })
    
    # Ordena por probabilidade
    df = df.sort_values('proba', ascending=False).reset_index(drop=True)
    
    # Calcula taxas acumuladas
    df['cumulative_bad'] = df['target'].cumsum() / df['target'].sum()
    df['cumulative_good'] = (1 - df['target']).cumsum() / (1 - df['target']).sum()
    df['ks'] = abs(df['cumulative_bad'] - df['cumulative_good'])
    
    # Encontra KS máximo
    ks_max = df['ks'].max()
    idx_max = df['ks'].idxmax()
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(df.index, df['cumulative_bad'], color='red', lw=2, label='Cumulative Bad Rate')
    ax.plot(df.index, df['cumulative_good'], color='green', lw=2, label='Cumulative Good Rate')
    ax.plot(df.index, df['ks'], color='blue', lw=2, label='KS Statistic')
    
    # Marca o KS máximo
    ax.axvline(idx_max, color='orange', linestyle='--', lw=2, label=f'Max KS = {ks_max:.4f}')
    ax.scatter(idx_max, ks_max, color='orange', s=100, zorder=5)
    
    ax.set_xlabel('Population (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Kolmogorov-Smirnov (KS) Plot - Max KS = {ks_max:.4f}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo = GRAPHICS_DIR / f'ks_plot_{timestamp}.png'
    plt.savefig(arquivo, bbox_inches='tight')
    print(f'✓ Salvo: {arquivo}')
    plt.close()


def plotar_precision_recall_curve(y_true, proba):
    """
    Gera curva Precision-Recall
    """
    print('\n📈 Gerando Precision-Recall Curve...')
    
    # Calcula precision-recall
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    avg_precision = average_precision_score(y_true, proba)
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, color='steelblue', lw=2, 
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.axhline(y=y_true.mean(), color='gray', linestyle='--', lw=2, 
               label=f'Baseline (No Skill = {y_true.mean():.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo = GRAPHICS_DIR / f'precision_recall_curve_{timestamp}.png'
    plt.savefig(arquivo, bbox_inches='tight')
    print(f'✓ Salvo: {arquivo}')
    plt.close()


def gerar_todos_graficos():
    """
    Gera todos os gráficos de análise
    """
    print('\n' + '='*70)
    print('🎨 GERAÇÃO DE GRÁFICOS - MODELO XGBOOST')
    print('='*70)
    
    try:
        # Carrega dados e modelo
        X, y, proba, pred, modelo = carregar_dados_e_modelo()
        
        # Gera todos os gráficos
        plotar_feature_importance(modelo, X.columns.tolist(), top_n=20)
        plotar_curva_roc(y, proba)
        plotar_distribuicao_scores(y, proba, THRESHOLD)
        plotar_confusion_matrix(y, pred)
        plotar_ks_plot(y, proba)
        plotar_precision_recall_curve(y, proba)
        
        print('\n' + '='*70)
        print('✅ TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO')
        print(f'📁 Salvos em: {GRAPHICS_DIR}')
        print('='*70)
        
    except Exception as e:
        print(f'\n❌ ERRO: {str(e)}')
        import traceback
        traceback.print_exc()
        raise


# ================================================================================
# EXECUÇÃO PRINCIPAL

if __name__ == '__main__':
    gerar_todos_graficos()
