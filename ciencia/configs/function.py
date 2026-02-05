import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score


def calcular_ks_statistic(y_true, y_score):
    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values(by='score', ascending=False)
    total_events = df.target.sum()
    total_non_events = len(df) - total_events
    df['cum_events'] = df.target.cumsum()
    df['cum_non_events'] = (df.target == 0).cumsum()
    df['cum_events_percent'] = df.cum_events / total_events
    df['cum_non_events_percent'] = df.cum_non_events / total_non_events
    ks_statistic = np.abs(df.cum_events_percent - df.cum_non_events_percent).max()
    return ks_statistic


def avaliar_modelo(X_train, y_train, X_test, y_test, modelo, nm_modelo):

    feature_names = list(X_train.columns)
    # Criação da figura e dos eixos
    fig, axs = plt.subplots(5, 2, figsize=(15, 30))  # Ajustado para incluir novos gráficos
    plt.tight_layout(pad=6.0)

    # Cor azul claro
    cor = 'skyblue'

    # Taxa de Evento e Não Evento
    event_rate = np.mean(y_train)
    non_event_rate = 1 - event_rate
    axs[0, 0].bar(['Evento', 'Não Evento'], [event_rate, non_event_rate], color=[cor, 'lightcoral'])
    axs[0, 0].set_title('Taxa de Evento e Não Evento')
    axs[0, 0].set_ylabel('Proporção')

    # Importância dos Atributos
    importancias = None
    if hasattr(modelo, 'coef_'):
        importancias = np.abs(modelo.coef_[0])
    elif hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_

    if importancias is not None:
        importancias_df = pd.DataFrame({'feature': feature_names, 'importance': importancias})
        importancias_df = importancias_df.sort_values(by='importance', ascending=True)

        axs[0, 1].barh(importancias_df['feature'], importancias_df['importance'], color=cor)
        axs[0, 1].set_title('Importância das Variáveis - ' + nm_modelo)
        axs[0, 1].set_xlabel('Importância')

    else:
        axs[0, 1].axis('off')  # Desativa o subplot se não houver importâncias para mostrar

    # Confusion Matrix - Treino
    y_pred_train = modelo.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    axs[1, 0].imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Confusion Matrix - Treino - ' + nm_modelo)
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xticklabels(['0', '1'])
    axs[1, 0].set_yticklabels(['0', '1'])
    thresh = cm_train.max() / 2.
    for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
        axs[1, 0].text(j, i, format(cm_train[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_train[i, j] > thresh else "black")

    # Confusion Matrix - Teste
    y_pred_test = modelo.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    axs[1, 1].imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 1].set_title('Confusion Matrix - Teste - ' + nm_modelo)
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_xticklabels(['0', '1'])
    axs[1, 1].set_yticklabels(['0', '1'])
    thresh = cm_test.max() / 2.
    for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
        axs[1, 1].text(j, i, format(cm_test[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_test[i, j] > thresh else "black")

    # ROC Curve - Treino e Teste
    y_score_train = modelo.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
    axs[2, 0].plot(fpr_train, tpr_train, color=cor, label='Treino')

    y_score_test = modelo.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
    axs[2, 0].plot(fpr_test, tpr_test, color='darkorange', label='Teste')

    axs[2, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axs[2, 0].set_title('ROC Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 0].set_xlabel('False Positive Rate')
    axs[2, 0].set_ylabel('True Positive Rate')
    axs[2, 0].legend(loc="lower right")

    # Precision-Recall Curve - Treino e Teste
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
    axs[2, 1].plot(recall_train, precision_train, color=cor, label='Treino')

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_score_test)
    axs[2, 1].plot(recall_test, precision_test, color='darkorange', label='Teste')

    axs[2, 1].set_title('Precision-Recall Curve - Treino e Teste - ' + nm_modelo)
    axs[2, 1].set_xlabel('Recall')
    axs[2, 1].set_ylabel('Precision')
    axs[2, 1].legend(loc="upper right")

    # Gini - Treino e Teste
    auc_train = roc_auc_score(y_train, y_score_train)
    gini_train = 2 * auc_train - 1
    auc_test = roc_auc_score(y_test, y_score_test)
    gini_test = 2 * auc_test - 1
    axs[3, 0].bar(['Treino', 'Teste'], [gini_train, gini_test], color=[cor, 'darkorange'])
    axs[3, 0].set_title('Gini - ' + nm_modelo)
    axs[3, 0].set_ylim(0, 1)
    axs[3, 0].text('Treino', gini_train + 0.01, f'{gini_train:.2f}', ha='center', va='bottom')
    axs[3, 0].text('Teste', gini_test + 0.01, f'{gini_test:.2f}', ha='center', va='bottom')

    # KS - Treino e Teste
    ks_train = calcular_ks_statistic(y_train, y_score_train)
    ks_test = calcular_ks_statistic(y_test, y_score_test)
    axs[3, 1].bar(['Treino', 'Teste'], [ks_train, ks_test], color=[cor, 'darkorange'])
    axs[3, 1].set_title('KS - ' + nm_modelo)
    axs[3, 1].set_ylim(0, 1)
    axs[3, 1].text('Treino', ks_train + 0.01, f'{ks_train:.2f}', ha='center', va='bottom')
    axs[3, 1].text('Teste', ks_test + 0.01, f'{ks_test:.2f}', ha='center', va='bottom')


    # Decile Analysis - Teste
    scores = modelo.predict_proba(X_test)[:, 1]
    noise = np.random.uniform(0, 0.0001, size=scores.shape)  # Adiciona um pequeno ruído
    scores += noise
    deciles = pd.qcut(scores, q=10, duplicates='drop')
    decile_analysis = y_test.groupby(deciles).mean()
    axs[4, 1].bar(range(1, len(decile_analysis) + 1), decile_analysis, color='darkorange')
    axs[4, 1].set_title('Ordenação do Score - Teste - ' + nm_modelo)
    axs[4, 1].set_xlabel('Faixas de Score')
    axs[4, 1].set_ylabel('Taxa de Evento')

    # Decile Analysis - Treino
    scores_train = modelo.predict_proba(X_train)[:, 1]
    noise = np.random.uniform(0, 0.0001, size=scores_train.shape)  # Adiciona um pequeno ruído
    scores_train += noise
    deciles_train = pd.qcut(scores_train, q=10, duplicates='drop')
    decile_analysis_train = y_train.groupby(deciles_train).mean()
    axs[4, 0].bar(range(1, len(decile_analysis_train) + 1), decile_analysis_train, color=cor)
    axs[4, 0].set_title('Ordenação do Score - Treino - ' + nm_modelo)
    axs[4, 0].set_xlabel('Faixas de Score')
    axs[4, 0].set_ylabel('Taxa de Evento')

    # Mostrar os gráficos
    plt.show()

#=======================================================================================================
def plot_by_safra(dataframe, target, explicativa, safra):

  df_copy = dataframe.copy()
  # Ou conversão direta se já for um código/número
  df_copy['SAFRA'] = df_copy['SAFRA'].astype(str)

  # Se a variável explicativa for numérica, arredonda para 4 casas decimais e converte para string
  if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
      df_copy[explicativa] = df_copy[explicativa].apply(lambda x: round(x, 4)).astype(str)

  # Calcula a taxa de evento e o volume por safra e categoria da variável explicativa
  result = df_copy.groupby([safra, explicativa]).agg({target: 'mean', explicativa: 'count'}).rename(columns={explicativa: 'Volume'}).reset_index()

  # Plota o gráfico
  fig, ax1 = plt.subplots(figsize=(12, 6))

  # Eixo Y esquerdo: Volume total por safra
  volume_by_safra = result.groupby(safra).agg({'Volume': 'sum'}).reset_index()
  bars = ax1.bar(volume_by_safra[safra], volume_by_safra['Volume'], color='lightblue', label='Volume Total (Barras)')
  ax1.set_xlabel('Safra')
  ax1.set_ylabel('Volume', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # Eixo Y direito: Taxa de Evento por categoria
  ax2 = ax1.twinx()
  for category in result[explicativa].unique():
      subset = result[result[explicativa] == category]
      ax2.plot(subset[safra], subset[target] * 100, marker='o', linestyle='-', label=f'Taxa de Evento ({category})')
  ax2.set_ylabel('Taxa de Evento (%)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  plt.title(f'Volume Total e Taxa de Evento por {explicativa} ao longo das Safras')
  plt.legend(loc='upper left')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

#=======================================================================================================
def calculate_woe_iv(df, feature, target):
    """
    Calcula WOE (Weight of Evidence) e IV (Information Value) para uma variável.
    """
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': df[df[feature] == val].count()[feature],
            'Good': df[(df[feature] == val) & (df[target] == 1)].count()[feature],
            'Bad': df[(df[feature] == val) & (df[target] == 0)].count()[feature]
        })

    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()

    return iv

def iv_table(df, target):
    """
    Retorna uma tabela com IV para todas as variáveis em relação ao target.
    """
    iv_list = []
    for col in df.columns:
        if col == target:
            continue
        iv = calculate_woe_iv(df, col, target)
        if iv < 0.02:
            predictiveness = 'Inútil para a predição'
        elif iv < 0.1:
            predictiveness = 'Preditor Fraco'
        elif iv < 0.3:
            predictiveness = 'Preditor Moderado'
        else:
            predictiveness = 'Preditor Forte'
        iv_list.append({
            'Variável': col,
            'IV': iv,
            'Preditividade': predictiveness
        })

    return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)

#=======================================================================================================
# avalia transformações matemáticas via R² do log-odds para decisão de feature engineering
def calculate_r2_for_logodds(df, variables, target, threshold):
    results = []

    transformations = {
        'Log': lambda x: np.log(x + 1),
        'Quadratic': lambda x: x ** 2,
        'Square Root': lambda x: np.sqrt(np.clip(x, 0, None)),
        'Reciprocal': lambda x: 1 / (x + 1)
    }

    for variable in variables:
        best_r2 = -np.inf
        best_transform = 'None'

        for name, func in transformations.items():
            x = func(df[variable])

            temp = pd.DataFrame({
                'x': x,
                'y': df[target]
            })

            temp['bin'] = safe_qcut(temp['x'])

            if temp['bin'] is None:
                continue

            stats = temp.groupby('bin').agg(
                x_mean=('x', 'mean'),
                y_mean=('y', 'mean')
            ).dropna()

            stats = stats[(stats['y_mean'] > 0) & (stats['y_mean'] < 1)]

            if len(stats) < 3:
                continue

            log_odds = np.log(stats['y_mean'] / (1 - stats['y_mean']))

            model = LinearRegression().fit(stats[['x_mean']], log_odds)
            r2 = model.score(stats[['x_mean']], log_odds)

            if r2 > best_r2:
                best_r2 = r2
                best_transform = name

        feat_eng = "Usar como contínua" if best_r2 >= threshold else "Categorizar"

        results.append({
            'Variable': variable,
            'Best Transformation': best_transform,
            'R² Log-Odds': best_r2,
            'Feat Eng': feat_eng,
            'Transformation Equation': f'{best_transform}({variable})'
        })

    return pd.DataFrame(results)

#=======================================================================================================
# propósito: tornar o qcut robusto contra NaN, inf e bins degenerados
def safe_qcut(series, max_bins=10):
    s = series.replace([np.inf, -np.inf], np.nan).dropna()

    if s.nunique() < 3:
        return None  # sem sinal estatístico

    n_bins = min(max_bins, s.nunique())

    try:
        return pd.qcut(s, q=n_bins, duplicates='drop')
    except ValueError:
        # plano B: cut com bins uniformes
        return pd.cut(s, bins=n_bins, duplicates='drop')
    
#=======================================================================================================
# propósito: avaliar transformações via R² do log-odds de forma estável
def calculate_r2_for_logodds_and_transformations(df, variables, target, threshold):
    results = []

    transformations = {
        'Log': lambda x: np.log(x + 1),
        'Quadratic': lambda x: x ** 2,
        'Square Root': lambda x: np.sqrt(np.clip(x, 0, None)),
        'Reciprocal': lambda x: 1 / (x + 1)
    }

    for variable in variables:
        best_r2 = -np.inf
        best_transform = 'None'

        for name, func in transformations.items():
            x = func(df[variable])

            temp = pd.DataFrame({
                'x': x,
                'y': df[target]
            })

            temp['bin'] = safe_qcut(temp['x'])

            if temp['bin'] is None:
                continue

            stats = temp.groupby('bin').agg(
                x_mean=('x', 'mean'),
                y_mean=('y', 'mean')
            ).dropna()

            stats = stats[(stats['y_mean'] > 0) & (stats['y_mean'] < 1)]

            if len(stats) < 3:
                continue

            log_odds = np.log(stats['y_mean'] / (1 - stats['y_mean']))

            model = LinearRegression().fit(stats[['x_mean']], log_odds)
            r2 = model.score(stats[['x_mean']], log_odds)

            if r2 > best_r2:
                best_r2 = r2
                best_transform = name

        feat_eng = "Usar como contínua" if best_r2 >= threshold else "Categorizar"

        results.append({
            'Variable': variable,
            'Best Transformation': best_transform,
            'R² Log-Odds': best_r2,
            'Feat Eng': feat_eng,
            'Transformation Equation': f'{best_transform}({variable})'
        })

    return pd.DataFrame(results)


#=======================================================================================================
# aplica transformações contínuas previamente definidas
def apply_transformations_from_map(df, transform_map, drop_original=False):
    transformed_df = df.copy()

    transformations = {
        'Log': lambda x: np.log(x + 1),
        'Quadratic': lambda x: x ** 2,
        'Square Root': lambda x: np.sqrt(np.clip(x, 0, None)),
        'Reciprocal': lambda x: 1 / (x + 1)
    }

    for var, transform_name in transform_map.items():
        if transform_name in transformations:
            transformed_df[f'TFE_{var}'] = transformations[transform_name](transformed_df[var])

    if drop_original:
        transformed_df.drop(columns=list(transform_map.keys()), inplace=True)

    return transformed_df

print("Funções extras carregadas com sucesso")