# Funções para manipulação de dados e análise
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def basic_information(df):
    """Informações básicas do dataset"""
    print(f'Dataset carregado com {df.shape[0]:,} registros e {df.shape[1]:,} colunas')
    print(f'Colunas: {df.columns.tolist()}')


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
        elif 'int' in tipo.lower():
            tipo_real = '🔢 Número Inteiro'
        elif 'float' in tipo:
            tipo_real = '🔢 Número Decimal'
        elif 'date' in tipo:
            tipo_real = '📅 Data'  
        else:
            tipo_real = f'❓ {tipo}'
    
        print(f'{col:30} → {tipo_real}')


def dataset_info_table(df):
    """Tabela com informações do dataset"""
    tabela = pd.DataFrame({
        'variavel': df.columns,
        'tipo': df.dtypes.astype(str),
        'n_nulos': df.isna().sum(),
        'Nulos_%': (df.isna().mean() * 100).round(2),
        'cardinalidade': df.nunique()
    }).reset_index(drop=True)

    tabela['tipo'] = tabela['tipo'].replace({
        'object': 'categorica',
        'category': 'categorica',
        'int64': 'numerica',
        'Int64': 'numerica',
        'int32': 'numerica',
        'float64': 'numerica',
        'datetime64[ns]': 'data'
    })
    return tabela


def check_duplicates(df):
    """Verifica duplicatas no dataset"""
    duplicatas_totais = df.duplicated().sum()
    print(f'Registros duplicados: {duplicatas_totais:,}')

    if duplicatas_totais > 0:
        pct_duplicatas = (duplicatas_totais / len(df)) * 100
        print(f'Percentual de duplicatas: {pct_duplicatas:.2f}%')
        print('⚠️ Atenção: Há registros duplicados que podem precisar ser removidos\n')
    else:
        print('✅ Nenhuma duplicata encontrada')


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

    return missing_df


def show_dataframe_samples(df, n=3):
    """Mostra amostras do dataframe"""
    print(f'\n PRIMEIRAS {n} LINHAS:')
    display(df.head(n))

    print(f'\n ÚLTIMAS {n} LINHAS:')
    display(df.tail(n))

    print('\n AMOSTRA ALEATÓRIA:')
    display(df.sample(min(n, len(df))))


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


def plot_numerical_distributions(df, numerical_cols=None, n_cols=3):
    """Plota distribuições de variáveis numéricas"""
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limitar a 20 variáveis para não sobrecarregar
    numerical_cols = numerical_cols[:20]
    
    n_vars = len(numerical_cols)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, col in enumerate(numerical_cols):
        axes[i].hist(df[col].dropna(), bins=50, edgecolor='black')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')
    
    # Remover subplots vazios
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# preenche nulos com media ou mediana conforme parametro
def custom_fillna(df, strategy='median'):
    numerical_cols = df.select_dtypes(
        include=['float64', 'float32', 'int64', 'int32']
    ).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    stats = {
        'numerical': {},
        'categorical_fill': 'Desconhecido'
    }

    for col in numerical_cols:
        stats['numerical'][col] = (
            df[col].median() if strategy == 'median' else df[col].mean()
        )
        df[col] = df[col].fillna(stats['numerical'][col])

    df[categorical_cols] = df[categorical_cols].fillna(stats['categorical_fill'])

    return df, stats



print("Funções básicas carregadas com sucesso")
