from __future__ import annotations

from pyspark.sql import SparkSession

from llm_spark_pipeline.config import Settings


def get_spark(settings: Settings) -> SparkSession:
    builder = (
        SparkSession.builder.appName(settings.spark_app_name)
        .master(settings.spark_master)
    )

    if settings.aws_access_key_id and settings.aws_secret_access_key:
        builder = builder.config("spark.hadoop.fs.s3a.access.key", settings.aws_access_key_id)
        builder = builder.config("spark.hadoop.fs.s3a.secret.key", settings.aws_secret_access_key)
    if settings.aws_default_region:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint.region", settings.aws_default_region)

    return builder.getOrCreate()

