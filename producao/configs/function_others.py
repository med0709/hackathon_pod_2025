import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# Teste oficial de linearidade no logit via R²
def logit_linearity_test(df, feature, target, q=5):
    bins = pd.qcut(df[feature], q=q, duplicates='drop')

    grouped = df.groupby(bins)[target].agg(['mean', 'count'])
    grouped['p'] = grouped['mean'].clip(0.001, 0.999)
    grouped['log_odds'] = np.log(grouped['p'] / (1 - grouped['p']))

    x = df.groupby(bins)[feature].mean().loc[grouped.index].values.reshape(-1, 1)
    y = grouped['log_odds'].values

    if len(y) < 3:
        return np.nan

    r2 = LinearRegression().fit(x, y).score(x, y)
    return r2

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
# Criar faixas ótimas usando o target
def decision_tree_binning(df, variable, target, n_bins=5):
    X = df[[variable]]
    y = df[target]

    tree = DecisionTreeClassifier(max_leaf_nodes=n_bins)
    tree.fit(X, y)

    leaf = tree.apply(X)

    temp = pd.DataFrame({
        variable: df[variable],
        'Leaf': leaf,
        target: y
    })

    bins = temp.groupby('Leaf').agg({
        target: 'mean',
        variable: ['count', 'min', 'max']
    }).reset_index()

    bins.columns = ['Leaf', 'EventRate', 'Volume', 'Lower', 'Upper']
    bins = bins.sort_values('Lower')

    bins.iloc[0, bins.columns.get_loc('Lower')] = -np.inf
    bins.iloc[-1, bins.columns.get_loc('Upper')] = np.inf

    return bins

#====================================================================================================
# objetivo: converter safra para string e manter o gráfico original com legenda condicional
def plot_by_safra(dataframe, target, explicativa, safra):

    import pandas as pd
    import matplotlib.pyplot as plt

    df_copy = dataframe.copy()

    # Converte a safra para string antes de qualquer agrupamento ou plot
    df_copy[safra] = df_copy[safra].astype(str)

    # Se a variável explicativa for numérica, arredonda para 4 casas decimais e converte para string
    if pd.api.types.is_numeric_dtype(df_copy[explicativa]):
        df_copy[explicativa] = df_copy[explicativa].round(4).astype(str)

    # Calcula a taxa de evento e o volume por safra e categoria da variável explicativa
    result = df_copy.groupby([safra, explicativa]).agg({target: "mean", explicativa: "count"}).rename(columns={explicativa: "Volume"}).reset_index()

    # Cria a figura do gráfico
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eixo Y esquerdo: Volume total por safra
    volume_by_safra = result.groupby(safra).agg({"Volume": "sum"}).reset_index()

    ax1.bar(volume_by_safra[safra], volume_by_safra["Volume"], label="Volume Total (Barras)")
    ax1.set_xlabel("Safra")
    ax1.set_ylabel("Volume")
    ax1.tick_params(axis="y")

    # Eixo Y direito: Taxa de Evento por categoria
    ax2 = ax1.twinx()

    for category in result[explicativa].unique():
        subset = result[result[explicativa] == category]
        ax2.plot(subset[safra], subset[target] * 100, marker="o", linestyle="-", label=f"Taxa de Evento ({category})")

    ax2.set_ylabel("Taxa de Evento (%)")
    ax2.tick_params(axis="y")

    # Exibe a legenda apenas se houver menos de 15 categorias
    if result[explicativa].nunique() < 15:
        fig.legend(loc="upper left")

    # Título e ajustes finais do gráfico
    plt.title(f"Volume Total e Taxa de Evento por {explicativa} ao longo das Safras")
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


print("Funções extras carregadas com sucesso")