# proposito: gerar tabela vertical (Treino/Teste por algoritmo) com AUC, GINI e KS

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

# ================================================================================================
def ks_from_proba(y_true, y_proba_pos):
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    return np.max(tpr - fpr)

# ================================================================================================
def metrics_block(y_true, y_pred, y_proba_pos):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba_pos)
    gini = 2*auc - 1
    ks = ks_from_proba(y_true, y_proba_pos)
    return acc, prec, rec, auc, gini, ks

# ================================================================================================
def build_metrics_table(models_list, X_train, y_train, X_test, y_test, finalize_fn):
    rows = []

    for mdl in models_list:
        name = type(mdl).__name__
        final_mdl = finalize_fn(mdl)

        # Treino
        proba_tr = final_mdl.predict_proba(X_train)[:, 1]
        pred_tr  = (proba_tr >= 0.5).astype(int)
        acc, prec, rec, auc, gini, ks = metrics_block(y_train.values, pred_tr, proba_tr)
        rows.append([name, 'Treino', acc, prec, rec, auc, gini, ks])

        # Teste
        proba_te = final_mdl.predict_proba(X_test)[:, 1]
        pred_te  = (proba_te >= 0.5).astype(int)
        acc, prec, rec, auc, gini, ks = metrics_block(y_test.values, pred_te, proba_te)
        rows.append([name, 'Teste', acc, prec, rec, auc, gini, ks])

    df = pd.DataFrame(
        rows,
        columns=['Algoritmo','Conjunto','Acuracia','Precisao','Recall','AUC_ROC','GINI','KS']
    )

    # ordenação executiva: por algoritmo e depois Treino/Teste
    ordem_conjunto = pd.CategoricalDtype(['Treino','Teste'], ordered=True)
    df['Conjunto'] = df['Conjunto'].astype(ordem_conjunto)
    df = df.sort_values(['Algoritmo','Conjunto']).reset_index(drop=True)

    return df

# ================================================================================================
print("Funções pycaret carregadas com sucesso")