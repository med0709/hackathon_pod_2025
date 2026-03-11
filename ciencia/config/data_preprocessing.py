# funções para análise dos dados

import pandas as pd
#=======================================================================================================
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
# Imputar nulos e registrar estatísticas para produção
def custom_fillna(df, strategy='median'):
    # Seleciona todas as colunas numéricas, incluindo Int64 nullable
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    stats = {
        'numerical': {},
        'categorical_fill': 'Desconhecido',
        'categorical_cols': categorical_cols
    }

    # Imputação numérica
    for col in numerical_cols:
        stats['numerical'][col] = df[col].median() if strategy == 'median' else df[col].mean()
        df[col] = df[col].fillna(stats['numerical'][col])

    # Imputação categórica
    df[categorical_cols] = df[categorical_cols].fillna(stats['categorical_fill'])

    return df, stats

#=======================================================================================================
print("Funções extras carregadas com sucesso")