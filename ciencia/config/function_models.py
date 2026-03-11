# Higienizar categorias "desconhecido", tipar inteiros nullable e cap em SCORE_RATEO
import numpy as np
import pandas as pd

def sanitize_dataframe(
    df: pd.DataFrame,
    unknown_pattern: str = r'(?i)^desconhecido$',
    score_col: str = 'SCORE_RATEO',
    score_min: float = 0,
    score_max: float = 10,
    convert_to_nullable_int: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    - Normaliza valores 'desconhecido' (qualquer capitalização/espaço) para NaN em colunas categóricas
    - Opcional: converte colunas não numéricas para Int64 (nullable)
    - Aplica cap em coluna de score para robustez
    """
    data = df if inplace else df.copy()

    # Normalizar "desconhecido" -> NaN em colunas object/categóricas
    cols_obj = data.select_dtypes(include=['object', 'category']).columns
    for col in cols_obj:
        s = data[col].astype(str).str.strip()
        if s.str.match(unknown_pattern).any():
            data[col] = s.replace(unknown_pattern, np.nan, regex=True)

    # Converter não numéricas para Int64 (nullable), se fizer sentido no teu schema
    if convert_to_nullable_int:
        cols_nao_numericas = data.select_dtypes(exclude=['float', 'int']).columns
        for col in cols_nao_numericas:
            # tenta converter de forma segura
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')

    # Cap no score para evitar explosões no modelo
    if score_col in data.columns:
        data[score_col] = data[score_col].clip(score_min, score_max)

    return data

# ================================================================================================
# aplicar preenchimento de missing numéricos e categóricos em múltiplos datasets
def preencher_missing_com_stats(df, stats):
    df = df.copy()

    # numéricas
    for col, value in stats['numerical'].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(stats['categorical_fill'])

    return df


