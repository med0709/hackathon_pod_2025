import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# mapear CEP (3 dígitos) para macro-região com tratamento de valores inválidos
def cep_3_para_regiao(cep_3):
    try:
        c = str(cep_3).zfill(3)

        # garante que são só dígitos
        if not c.isdigit():
            return 'DESCONHECIDO'

        prefixo_2 = int(c[:2])
        d1 = int(c[0])

        # exceções geográficas
        if 64 <= prefixo_2 <= 65: return 'NORDESTE'   # Maranhão
        if prefixo_2 == 77:       return 'NORTE'      # Tocantins
        if prefixo_2 == 69:       return 'NORTE'      # AM / RR / AC / RO
        if d1 == 7 and 40 <= prefixo_2 <= 49:
            return 'NORDESTE'                         # BA / SE

        mapping_d1 = {
            0: 'SUDESTE', 1: 'SUDESTE', 2: 'SUDESTE', 3: 'SUDESTE',
            4: 'SUL', 5: 'SUL',
            6: 'NORTE',
            7: 'CENTRO_OESTE',
            8: 'NORDESTE',
            9: 'NORDESTE'
        }

        return mapping_d1.get(d1, 'DESCONHECIDO')

    except Exception:
        return 'DESCONHECIDO'

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
def calculate_r2_for_logodds(df, variables, target, threshold):
    results = []

    for variable in variables:
        # Verificando o número de valores únicos
        unique_vals = df[variable].nunique()
        if unique_vals == 1:
            print(f"{variable} tem apenas um valor único. Ignorando...")
            continue

        n_bins = min(10, unique_vals)

        # Criando bins para a variável
        df['bin'] = pd.cut(df[variable], bins=n_bins, labels=False, duplicates='drop')

        # Calculando a proporção de eventos positivos para cada bin
        mean_target = df.groupby('bin')[target].mean()

        # Calculando o log(odds) e tratando valores infinitos
        log_odds = np.log(mean_target / (1 - mean_target)).replace([np.inf, -np.inf], np.nan).dropna()

        # Calculando R^2
        X = df.groupby('bin')[variable].mean()[log_odds.index].values.reshape(-1, 1)
        y = log_odds.values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        # Decidindo sobre a engenharia de recursos com base no valor de R^2 e no threshold fornecido
        feat_eng = "Usar como contínua" if r2 > threshold else "Categorizar"

        results.append({
            'Variable': variable,
            'R^2': r2,
            'Feat Eng': feat_eng
        })

        # Removendo a coluna bin
        df.drop('bin', axis=1, inplace=True)

    return pd.DataFrame(results)

#=======================================================================================================
# avalia transformações matemáticas via R² do log-odds para decisão de feature engineering
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

        x_raw = df[variable]

        for name, func in transformations.items():
            x = func(x_raw)

            if x.isna().all() or x.nunique() <= 1:
                continue

            temp = pd.DataFrame({
                'x': x,
                'y': df[target]
            }).dropna()

            n_bins = min(10, temp['x'].nunique())
            temp['bin'] = pd.qcut(temp['x'], q=n_bins, duplicates='drop')

            stats = temp.groupby('bin').agg(
                x_mean=('x', 'mean'),
                y_mean=('y', 'mean')
            )

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