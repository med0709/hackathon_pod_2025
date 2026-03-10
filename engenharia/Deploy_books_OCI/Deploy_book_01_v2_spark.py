# -*- coding: utf-8 -*-

"""
Script ajustado para execução no Oracle DataFlow
"""

# =========================
# INICIAR SPARK
# =========================

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("book_variaveis_job").getOrCreate()

# =========================
# IMPORTS CIÊNCIA DE DADOS
# =========================

import pandas as pd
import numpy as np


# =========================
# CONEXÃO OCI
# =========================

pasta_in = 'base_score_bureau_movel_full/'
pasta_out = 'Feature_store/'

namespace = "@grxzqsiaote6/"

bucket_trusted = f"oci://TRUSTED{namespace}{pasta_in}"
bucket_feature_store = f"oci://BOOKS_VARIAVEIS{namespace}{pasta_out}"


print("Carregando dados do Object Storage...")

# =========================
# LEITURA DOS DADOS (SPARK)
# =========================

df_spark = spark.read.parquet(bucket_trusted)

# converter para pandas para usar o notebook
df_bureau = df_spark.toPandas()

print("Dados carregados com sucesso")


# =========================
# AJUSTE DOS TIPOS
# =========================

df_bureau['Ano'] = df_bureau['Ano'].astype('int')
df_bureau['Mes'] = df_bureau['Mes'].astype('int')
df_bureau['FLAG_INSTALACAO'] = df_bureau['FLAG_INSTALACAO'].astype('bool')
df_bureau['ProductDescription'] = df_bureau['ProductDescription'].astype('object')
df_bureau['ProductMigration'] = df_bureau['ProductMigration'].astype('object')
df_bureau['SCORE_01'] = df_bureau['SCORE_01'].astype('float32')
df_bureau['SCORE_02'] = df_bureau['SCORE_02'].astype('float32')
df_bureau['FPD'] = df_bureau['FPD'].astype('Int64')
df_bureau['NUM_CPF'] = df_bureau['NUM_CPF'].astype('object')
df_bureau['SAFRA'] = df_bureau['SAFRA'].astype('int')


# =========================
# FEATURE ENGINEERING
# =========================

df_bureau['SCORE_RATE'] = df_bureau['SCORE_02'] / df_bureau['SCORE_01']
df_bureau['SCORE_AVG'] = (df_bureau['SCORE_01'] + df_bureau['SCORE_02']) / 2
df_bureau['SCORE_DIFF'] = df_bureau['SCORE_02'] - df_bureau['SCORE_01']
df_bureau['SCORE_MIN'] = df_bureau[['SCORE_01', 'SCORE_02']].min(axis=1)


print("Feature engineering finalizada")


# =========================
# CONVERTER PARA SPARK
# =========================

df_final = spark.createDataFrame(df_bureau)


# =========================
# SALVAR RESULTADO
# =========================

print("Salvando parquet no Feature Store...")

df_final.write \
    .mode("overwrite") \
    .parquet(f"{bucket_feature_store}/book_variaveis_01")

print("Processo finalizado com sucesso")