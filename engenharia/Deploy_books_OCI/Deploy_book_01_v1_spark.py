# -*- coding: utf-8 -*-

"""
Feature Engineering - Book de Variáveis
Execução otimizada para Oracle DataFlow (Spark)
"""


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, least

spark = SparkSession.builder.appName("book_variaveis_job").getOrCreate()


# =========================
# CAMINHOS OCI
# =========================

pasta_in = "base_score_bureau_movel_full/"
pasta_out = "Feature_store/"

namespace = "@grxzqsiaote6/"

bucket_trusted = f"oci://TRUSTED{namespace}{pasta_in}"
bucket_feature_store = f"oci://BOOKS_VARIAVEIS{namespace}{pasta_out}"

print("Carregando dados do Object Storage...")


# =========================
# LEITURA DOS DADOS
# =========================

df_bureau = spark.read.parquet(bucket_trusted)

print("Dados carregados")


# =========================
# AJUSTE DOS TIPOS
# =========================

df_bureau = df_bureau \
    .withColumn("Ano", col("Ano").cast("int")) \
    .withColumn("Mes", col("Mes").cast("int")) \
    .withColumn("FLAG_INSTALACAO", col("FLAG_INSTALACAO").cast("boolean")) \
    .withColumn("ProductDescription", col("ProductDescription").cast("string")) \
    .withColumn("ProductMigration", col("ProductMigration").cast("string")) \
    .withColumn("SCORE_01", col("SCORE_01").cast("float")) \
    .withColumn("SCORE_02", col("SCORE_02").cast("float")) \
    .withColumn("FPD", col("FPD").cast("int")) \
    .withColumn("NUM_CPF", col("NUM_CPF").cast("string")) \
    .withColumn("SAFRA", col("SAFRA").cast("int"))


# =========================
# FEATURE ENGINEERING
# =========================

df_bureau = df_bureau \
    .withColumn("SCORE_RATE", col("SCORE_02") / col("SCORE_01")) \
    .withColumn("SCORE_AVG", (col("SCORE_01") + col("SCORE_02")) / 2) \
    .withColumn("SCORE_DIFF", col("SCORE_02") - col("SCORE_01")) \
    .withColumn("SCORE_MIN", least(col("SCORE_01"), col("SCORE_02")))

print("Features criadas")


# =========================
# SALVAR RESULTADO
# =========================

print("Salvando no Feature Store...")

df_bureau.write \
    .mode("overwrite") \
    .partitionBy("SAFRA","Ano","Mes") \
    .parquet(f"{bucket_feature_store}/book_variaveis_01")

print("Processamento finalizado com sucesso")