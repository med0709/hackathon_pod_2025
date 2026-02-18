# Funções para manipulação de dados e análise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

#=======================================================================================================
def basic_information(df):
    """Informações básicas do dataset"""
    print(f"Linhas: {df.shape[0]:,}")
    print(f"Colunas: {df.shape[1]:,}")
    colunas = df.columns.to_list()
    print(f"Colunas: {colunas}")

#=======================================================================================================
def data_type(df):
    """Análise dos tipos de dados"""
    tipos = df.dtypes.value_counts()
    for tipo, quantidade in tipos.items():
        print(f'{str(tipo):15}: {quantidade} variáveis')

    print(f'\n📊 Total: {len(df.columns)} variáveis')

    print('\n DETALHAMENTO POR VARIÁVEL:')
    for col in df.columns:
        tipo = str(df[col].dtype)
        if tipo == 'object':
            tipo_real = '📝 Texto/Categórica'
        elif tipo == 'str':
            tipo_real = '📝 Texto/Categórica'
        elif 'int' in tipo.lower():
            tipo_real = '🔢 Número Inteiro'
        elif 'float' in tipo:
            tipo_real = '🔢 Número Decimal'
        elif 'date' in tipo:
            tipo_real = '📅 Data'  
        else:
            tipo_real = f'❓ {tipo}'
    
        print(f'{col:30} → {tipo_real}')

#=======================================================================================================
def show_dataframe_samples(df, n=3):
    """Mostra amostras do dataframe"""
    print(f'\n PRIMEIRAS {n} LINHAS:')
    display(df.head(n))

    print(f'\n ÚLTIMAS {n} LINHAS:')
    display(df.tail(n))

    print('\n AMOSTRA ALEATÓRIA:')
    display(df.sample(min(n, len(df))))

#=======================================================================================================
def analyze_target(df, target_col='FPD'):
    """Análise da variável target"""
    print(f'\n📊 ANÁLISE DA VARIÁVEL TARGET: {target_col}')
    print(f'\nDistribuição:')
    counts = df[target_col].value_counts()
    for valor, freq in counts.items():
        pct = (freq / len(df)) * 100
        print(f'   {valor}: {freq:,} ({pct:.1f}%)')
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Countplot
    df[target_col].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title(f'Distribuição: {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Contagem')
    
    # Pie chart
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title(f'Proporção: {target_col}')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()

#=======================================================================================================
def analyze_missing_values(df, plot=True):
    """Analisa valores faltantes"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Variable': missing.index,
        'Missing_Values': missing.values,
        'Percentage': missing_pct.values
    })

    missing_df = missing_df.sort_values('Percentage', ascending=False)
    missing_df = missing_df[missing_df['Missing_Values'] > 0]

    if missing_df.empty:
        print('✅ Excellent! No missing values found')
        return missing_df

    print(f'\n⚠️ Missing values in {len(missing_df)} variables:\n')

    for _, row in missing_df.iterrows():
        print(f"{row['Variable']:30} → {row['Missing_Values']:,} ({row['Percentage']:.1f}%)")

    if plot and len(missing_df) <= 30:
        print()
        plt.figure(figsize=(10, 5))
        plt.bar(missing_df['Variable'][:30], missing_df['Percentage'][:30])
        plt.title('Percentage of Missing Values by Variable (Top 30)')
        plt.xlabel('Variables')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

#=======================================================================================================
def check_duplicates(df):
    """Verifica duplicatas no dataset"""
    duplicatas_totais = df.duplicated().sum()
    print(f'Registros duplicados: {duplicatas_totais:,}')

    if duplicatas_totais > 0:
        pct_duplicatas = (duplicatas_totais / len(df)) * 100
        display(df[df.duplicated()].head())
        print(f'Percentual de duplicatas: {pct_duplicatas:.2f}%')
        print('⚠️ Atenção: Há registros duplicados que podem precisar ser removidos\n')
    else:
        print('✅ Nenhuma duplicata encontrada')

#=======================================================================================================
# gera tabela de metadados do dataset para análise e governança
def info_table(df):
    num = df.select_dtypes(include='number')

    tabela = pd.DataFrame({
        "variavel": df.columns,
        "tipo": df.dtypes.astype(str),
        "n_nulos": df.isna().sum(),
        "Nulos_%": (df.isna().mean() * 100).round(2),
        "Cardinalidade": df.nunique(),
        "n_negativos": num.apply(lambda x: (x < 0).sum()).reindex(df.columns),
        "n_inf": num.apply(lambda x: np.isinf(x).sum()).reindex(df.columns),
        "inf_%": (num.apply(lambda x: np.isinf(x).mean()) * 100).round(2).reindex(df.columns),
        "espacos_vazios": df.select_dtypes("object")
            .apply(lambda x: (x.str.strip() == "").sum())
            .reindex(df.columns)
    }).reset_index(drop=True)

    # Ajustar tipo mais legível
    tabela["tipo"] = tabela["tipo"].replace({
        "object": "categorica",
        "category": "categorica",
        "int64": "numerica",
        "float64": "numerica",
        "Int64": "numerica",
        "datetime64[ns]": "data"
    })

    # ordenar SOMENTE se existir inf
    if tabela["n_inf"].sum() > 0:
        tabela = tabela.sort_values(by="n_inf", ascending=False).reset_index(drop=True)

    return tabela


def dataset_info_table(df, orderby="PC_nulos", ascending=False):
    tabela = pd.DataFrame({
        "Feature": df.columns,
        "QT_nulos": df.isna().sum(),
        "PC_nulos": (df.isna().mean() * 100).round(2),
        "QT_zeros": (df == 0).sum(numeric_only=True),
        "Cardinalidade": df.nunique(dropna=True),
        "Tipo_feature": df.dtypes.astype(str),
    }).reset_index(drop=True)

    if orderby is not None:
        tabela = tabela.sort_values(by=orderby, ascending=ascending).reset_index(drop=True)

    return tabela

#=======================================================================================================
def analyze_categorical_features(df, max_unique=20):
    """Análise de variáveis categóricas"""
    categoricas = df.select_dtypes(include=['object']).columns
    
    for col in categoricas:
        unique_count = df[col].nunique()
        print(f'\n🏷️ {col.upper()}:')
        print(f'   Valores únicos: {unique_count:,}')
        
        if unique_count <= max_unique:
            print('   Valores e frequências:')
            value_counts = df[col].value_counts()
            for valor, freq in value_counts.head(10).items():
                pct = (freq / len(df)) * 100
                print(f'      {valor}: {freq:,} ({pct:.1f}%)')
            
            if len(value_counts) > 10:
                print(f'      ... e mais {len(value_counts) - 10} valores')
        else:
            print(f'   ⚠️ Muitos valores únicos ({unique_count:,}) - variável de alta cardinalidade')

#=======================================================================================================
# explores numerical variables
def analyze_numerical_features(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        print(f'\n📊 {col.upper()}:')

        stats = df[col].describe()
        print(f"   Minimum: {stats['min']:,.2f}")
        print(f"   Maximum: {stats['max']:,.2f}")
        print(f"   Mean: {stats['mean']:,.2f}")
        print(f"   Median: {stats['50%']:,.2f}")

        negatives = (df[col] < 0).sum()
        if negatives > 0:
            print(f'   ⚠️ Negative values: {negatives:,}')

        zeros = (df[col] == 0).sum()
        if zeros > 0:
            pct_zeros = (zeros / len(df)) * 100
            print(f'   🔵 Zero values: {zeros:,} ({pct_zeros:.1f}%)')

        unique_count = df[col].nunique()
        print(f'   🎯 Unique values: {unique_count:,}')

# ================================================================================================
# plots histogram and boxplot for numerical features in dual charts
def plot_numerical_dual_charts(df, numerical_cols):
    for col in numerical_cols:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram: {col}')

        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'Boxplot: {col}')

        plt.tight_layout()
        plt.show()

#=======================================================================================================
# decide automaticamente o tratamento de valores nulos apenas para colunas que possuem missing segundo o metadado
def decide_missing_treatment(
    df_dados,
    df_metadados,
    target_col,
    pc_nulos_col="PC_nulos",
    feature_col_meta="Feature",
    min_target_diff=0.05
):
    import pandas as pd

    report = []

    # filtra apenas colunas com missing no metadado
    meta_missing = df_metadados[df_metadados[pc_nulos_col] > 0]

    for _, row in meta_missing.iterrows():
        col = row[feature_col_meta]

        if col == target_col or col not in df_dados.columns:
            continue

        pc_nulos = row[pc_nulos_col]
        missing_pct = pc_nulos / 100 if pc_nulos > 1 else pc_nulos

        diff = abs(
            df_dados.loc[df_dados[col].isna(), target_col].mean()
            - df_dados.loc[df_dados[col].notna(), target_col].mean()
        )

        if missing_pct > 0.7:
            decision = "REMOVER_COLUNA"
        elif missing_pct > 0.2:
            decision = "MANTER_COM_FLAG" if diff >= min_target_diff else "REMOVER_COLUNA"
        elif missing_pct > 0.05:
            decision = "MANTER_COM_FLAG"
        elif missing_pct > 0.01:
            decision = "MANTER_COM_FLAG" if diff >= min_target_diff else "MANTER_SEM_FLAG"
        else:
            decision = "MANTER_SEM_FLAG"

        report.append({
            "feature": col,
            "percentual_missing": round(missing_pct, 4),
            "diferenca_target_missing": round(diff, 4),
            "decisao": decision
        })

    return pd.DataFrame(report)

#=======================================================================================================
# Imputar nulos e registrar estatísticas para produção
def custom_fillna(df, strategy='median'):
    numerical_cols = df.select_dtypes(
        include=['float64', 'float32', 'int64', 'int32']
    ).columns.tolist()

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    stats = {
        'numerical': {},
        'categorical_fill': 'Desconhecido',
        'categorical_cols': categorical_cols
    }

    for col in numerical_cols:
        stats['numerical'][col] = (
            df[col].median() if strategy == 'median' else df[col].mean()
        )
        df[col] = df[col].fillna(stats['numerical'][col])

    df[categorical_cols] = df[categorical_cols].fillna(stats['categorical_fill'])

    return df, stats


print("Funções básicas carregadas com sucesso")