# Importacao de pacotes

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

####################################
### Inicio do script de funcoes  ###
####################################

# Metadados referente ao conjunto de dados
def generate_metadata(dataframe):
    """
    Gera um dataframe contendo metadados das colunas do dataframe fornecido.

    :param dataframe: DataFrame para o qual os metadados serão gerados.
    :return: DataFrame contendo metadados.
    """

    # Coleta de metadados básicos
    metadata = pd.DataFrame({
        'nome_variavel': dataframe.columns,
        'tipo': dataframe.dtypes,
        'qt_nulos': dataframe.isnull().sum(),
        'percent_nulos': round((dataframe.isnull().sum() / len(dataframe))* 100,2),
        'cardinalidade': dataframe.nunique(),
    })
    metadata=metadata.sort_values(by='percent_nulos',ascending=False)
    metadata = metadata.reset_index(drop=True)

    return metadata

def drop_columns_high_missing(df, threshold=0.7, verbose=True):
    """
    Remove colunas com percentual de valores nulos maior que o threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de entrada
    threshold : float
        Percentual máximo permitido de valores nulos (default = 0.7)
    verbose : bool
        Se True, imprime colunas removidas

    Returns
    -------
    pandas.DataFrame
        DataFrame sem as colunas com alto missing
    """
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    if verbose:
        print(f"🧹 Colunas removidas (> {int(threshold*100)}% missing): {len(cols_to_drop)}")
        if cols_to_drop:
            for col in cols_to_drop:
                print(f" - {col}: {missing_ratio[col]:.1%}")

    return df.drop(columns=cols_to_drop)

def drop_single_cardinality_columns(df, verbose=True):
    """
    Remove colunas que possuem apenas um valor distinto (cardinalidade = 1).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de entrada
    verbose : bool
        Se True, imprime colunas removidas

    Returns
    -------
    pandas.DataFrame
        DataFrame sem colunas de cardinalidade 1
    """
    nunique = df.nunique(dropna=False)
    cols_to_drop = nunique[nunique == 1].index.tolist()

    if verbose:
        print(f"🧹 Colunas removidas (cardinalidade = 1): {len(cols_to_drop)}")
        for col in cols_to_drop:
            print(f" - {col}")

    return df.drop(columns=cols_to_drop)
