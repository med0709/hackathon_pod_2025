# Importacao de pacotes

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
from typing import List
from itertools import combinations
from typing import List


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


def calcular_idade_df(
    df: pd.DataFrame,
    col_nascimento: str,
    col_referencia: str,
    col_saida: str = 'IDADE'
) -> pd.DataFrame:
    """
    Calcula idade em anos completos e cria a coluna no DataFrame.
    
    Parâmetros:
    - df: DataFrame de entrada
    - col_nascimento: nome da coluna de data de nascimento
    - col_referencia: nome da coluna de data de referência (ex: safra)
    - col_saida: nome da coluna a ser criada (default = 'IDADE')
    
    Retorno:
    - DataFrame com a coluna de idade criada
    """

    df[col_saida] = (
        df[col_referencia].dt.year
        - df[col_nascimento].dt.year
        - (
            (df[col_referencia].dt.month < df[col_nascimento].dt.month)
            | (
                (df[col_referencia].dt.month == df[col_nascimento].dt.month)
                & (df[col_referencia].dt.day < df[col_nascimento].dt.day)
            )
        )
    ).astype("Int64")

    return df

def mapear_regiao_subregiao_texto(df: pd.DataFrame, coluna_cep3: str) -> pd.DataFrame:
    """
    A partir de CEP 3 dígitos, cria:
    - REGIAO_POSTAL
    - REGIAO_POSTAL_TXT
    - SUB_REGIAO_POSTAL
    """

    mapa_regiao = {
        '0': 'Grande São Paulo',
        '1': 'Interior de São Paulo',
        '2': 'Rio de Janeiro e Espírito Santo',
        '3': 'Minas Gerais',
        '4': 'Bahia e Sergipe',
        '5': 'Nordeste Oriental (PE, AL, PB, RN)',
        '6': 'Nordeste Setentrional (CE, PI, MA)',
        '7': 'Centro-Oeste e Norte',
        '8': 'Paraná e Santa Catarina',
        '9': 'Rio Grande do Sul'
    }

    cep3 = df[coluna_cep3].astype(str).str.zfill(3)

    # validação: apenas valores totalmente numéricos
    cep3_valido = cep3.str.isdigit()

    df['REGIAO_POSTAL'] = cep3.str[0]
    df['SUB_REGIAO_POSTAL'] = cep3.str[:2]
    df['REGIAO_POSTAL_TXT'] = df['REGIAO_POSTAL'].map(mapa_regiao)

    # qualquer valor não numérico vira Desconhecido
    df.loc[~cep3_valido, ['REGIAO_POSTAL', 'SUB_REGIAO_POSTAL', 'REGIAO_POSTAL_TXT']] = 'Desconhecido'

    # fallback extra caso o dígito seja numérico mas não esteja no mapa
    df['REGIAO_POSTAL_TXT'] = df['REGIAO_POSTAL_TXT'].fillna('Desconhecido')

    return df

def criar_flag_com_nome_do_valor(
    df: pd.DataFrame,
    colunas: List[str],
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Para colunas onde o preenchimento indica um perfil:
    - valor não nulo → 1
    - valor nulo → 0
    - nome da nova coluna = valor não nulo encontrado
    - remove a coluna original
    """

    for col in colunas:
        # obtém o valor não nulo (assumindo perfil único)
        valores = df[col].dropna().unique()

        if len(valores) == 0:
            # nenhuma informação na coluna
            continue

        if len(valores) > 1:
            raise ValueError(
                f"A coluna '{col}' possui mais de um valor distinto: {valores}"
            )

        nome_coluna = str(valores[0])

        df[nome_coluna] = df[col].notna().astype("Int8")

    if drop_original:
        df = df.drop(columns=colunas)

    return df


def find_duplicate_columns(df: pd.DataFrame):
    """
    Encontra colunas que são exatamente iguais (mesmos valores linha a linha).
    
    Retorna uma lista de tuplas:
    [(coluna_1, coluna_2), ...]
    """
    duplicated = []

    for col1, col2 in combinations(df.columns, 2):
        # equals trata NaN == NaN corretamente
        if df[col1].equals(df[col2]):
            duplicated.append((col1, col2))

    return duplicated

def convert_var_columns_to_numeric(
    df: pd.DataFrame,
    prefix: str = "var_",
    errors: str = "coerce",
    inplace: bool = False
):
    """
    Converte todas as colunas que começam com `prefix` para tipo numérico.

    Parâmetros:
    - df: DataFrame pandas
    - prefix: prefixo das colunas (default 'var_')
    - errors: comportamento do pandas.to_numeric ('coerce', 'ignore', 'raise')
    - inplace: se True, altera o df original

    Retorna:
    - DataFrame com colunas convertidas (ou None se inplace=True)
    """
    target_cols = [col for col in df.columns if col.startswith(prefix)]

    if not inplace:
        df = df.copy()

    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors=errors)

    return df if not inplace else None

def criar_coluna_safra(df: pd.DataFrame, coluna_datetime: str) -> pd.DataFrame:
    """
    Cria a coluna SAFRA no formato YYYYMM a partir de uma coluna datetime
    ou string no padrão 09OCT2023:00:00:00
    """

    df[coluna_datetime] = pd.to_datetime(
        df[coluna_datetime],
        format='%d%b%Y:%H:%M:%S',
        errors='coerce'
    )

    df['SAFRA'] = df[coluna_datetime].dt.strftime('%Y%m')

    return df


def criar_lags_por_safra(
    df: pd.DataFrame,
    col_cpf: str,
    col_safra: str,
    variaveis: List[str],
    janelas: List[int]
) -> pd.DataFrame:
    """
    Cria lags e acumulados temporais por CPF e SAFRA.

    Exemplo de janelas: [1, 3, 6]
    """

    df = df.copy()

    df[col_safra] = df[col_safra].astype(int)
    df = df.sort_values([col_cpf, col_safra])

    max_lag = max(janelas)

    for var in variaveis:
        # cria lags individuais até o máximo necessário
        for i in range(1, max_lag + 1):
            df[f'{var}_LAG_{i}'] = (
                df.groupby(col_cpf)[var].shift(i)
            )

        # cria acumulados para cada janela solicitada
        for janela in janelas:
            df[f'{var}_ULT_{janela}_SAFRAS'] = (
                df[[f'{var}_LAG_{i}' for i in range(1, janela + 1)]]
                .sum(axis=1)
            )

    return df
