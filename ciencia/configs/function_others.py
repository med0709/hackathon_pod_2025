# Funções auxiliares para modelagem e avaliação
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import pickle


def save_model(model, filepath):
    """Salva modelo em arquivo pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f'Modelo salvo em: {filepath}')


def load_model(filepath):
    """Carrega modelo de arquivo pickle"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f'Modelo carregado de: {filepath}')
    return model


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Matriz de Confusão'):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels or ['0', '1'],
                yticklabels=labels or ['0', '1'])
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.show()
    
    return cm


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Avalia modelo e retorna métricas"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    print('\n📊 MÉTRICAS DO MODELO:')
    for metric, value in metrics.items():
        print(f'   {metric:15}: {value:.4f}')
    
    return metrics


def plot_roc_curve(y_true, y_pred_proba, title='Curva ROC'):
    """Plota curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, title='Feature Importance'):
    """Plota importância das features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print('Modelo não possui feature importance')
        return
    
    # Criar dataframe
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feat_imp)), feat_imp['importance'])
    plt.yticks(range(len(feat_imp)), feat_imp['feature'])
    plt.xlabel('Importância')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feat_imp


def calculate_business_impact(y_true, y_pred, cost_per_line=50.0):
    """Calcula impacto financeiro das predições"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Custo: Falsos Negativos (aprovamos mas era mau pagador)
    # Benefício: Verdadeiros Positivos (aprovamos e não era mau pagador) 
    # Perda de oportunidade: Falsos Positivos (rejeitamos mas era bom pagador)
    
    # Assumindo que FPD=1 é mau pagador (inadimplente)
    # TP: predisse 0 (bom) e era 0 (bom) - ganhamos R$ 50
    # FP: predisse 0 (bom) e era 1 (mau) - perdemos R$ 50
    # TN: predisse 1 (mau) e era 1 (mau) - não perdemos nada
    # FN: predisse 1 (mau) e era 0 (bom) - perda de oportunidade R$ 50
    
    ganho = tn * cost_per_line  # Aprovamos bons pagadores
    perda_inadimplencia = fp * cost_per_line  # Aprovamos maus pagadores
    perda_oportunidade = fn * cost_per_line  # Rejeitamos bons pagadores
    
    resultado_liquido = ganho - perda_inadimplencia
    
    print('\n💰 IMPACTO FINANCEIRO:')
    print(f'   Ganho (aprovados corretos): R$ {ganho:,.2f}')
    print(f'   Perda (inadimplentes aprovados): R$ {perda_inadimplencia:,.2f}')
    print(f'   Perda de oportunidade (rejeitados incorretos): R$ {perda_oportunidade:,.2f}')
    print(f'   Resultado líquido: R$ {resultado_liquido:,.2f}')
    
    return {
        'ganho': ganho,
        'perda_inadimplencia': perda_inadimplencia,
        'perda_oportunidade': perda_oportunidade,
        'resultado_liquido': resultado_liquido
    }


print("Funções auxiliares carregadas com sucesso")
