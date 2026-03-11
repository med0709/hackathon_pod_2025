# funções para manipulação de dados e análise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def show_dataframe_samples(df, n=3):
    """Mostra amostras do dataframe"""
    print(f'\n PRIMEIRAS {n} LINHAS:')
    display(df.head(n))

    print(f'\n ÚLTIMAS {n} LINHAS:')
    display(df.tail(n))

    print('\n AMOSTRA ALEATÓRIA:')
    display(df.sample(min(n, len(df))))

#=======================================================================================================
def basic_information(df):
    """Informações básicas do dataset"""
    print(f"Linhas: {df.shape[0]:,} Colunas: {df.shape[1]:,}")
    colunas = df.columns.to_list()
    print(f"\nColunas: {colunas}")

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
    return tabela

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
        elif 'Float' in tipo:
            tipo_real = '📠 Número Decimal'
        elif 'float' in tipo:
            tipo_real = '📠 Número Decimal'
        elif 'date' in tipo:
            tipo_real = '📅 Data'  
        else:
            tipo_real = f'❓ {tipo}'
    
        print(f'{col:30} → {tipo_real}')

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
        print(f"{row['Variable']:35} → {row['Missing_Values']:,} ({row['Percentage']:.1f}%)")

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

print("Funções básicas carregadas com sucesso")