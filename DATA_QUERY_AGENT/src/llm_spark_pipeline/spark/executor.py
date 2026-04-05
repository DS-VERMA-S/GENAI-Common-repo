from __future__ import annotations

import logging
from typing import List

from pyspark.sql import DataFrame, functions as F

from llm_spark_pipeline.config import Settings
from llm_spark_pipeline.spark.session import get_spark
from llm_spark_pipeline.validation.schema import ActionPlan

logger = logging.getLogger(__name__)


def _read_input(spark, uri: str) -> DataFrame:
    if uri.endswith("/"):
        # assume partitioned parquet folder
        return spark.read.parquet(uri)
    if uri.endswith(".parquet"):
        return spark.read.parquet(uri)
    if uri.endswith(".csv"):
        return spark.read.option("header", "true").csv(uri)
    if uri.endswith(".json"):
        return spark.read.json(uri)
    return spark.read.parquet(uri)


def _apply_filters(df: DataFrame, filters) -> DataFrame:
    for f in filters:
        col = F.col(f.column)
        if f.operator == "=":
            df = df.filter(col == f.value)
        elif f.operator == "!=":
            df = df.filter(col != f.value)
        elif f.operator == ">":
            df = df.filter(col > f.value)
        elif f.operator == ">=":
            df = df.filter(col >= f.value)
        elif f.operator == "<":
            df = df.filter(col < f.value)
        elif f.operator == "<=":
            df = df.filter(col <= f.value)
        elif f.operator == "contains":
            df = df.filter(col.contains(f.value))
        elif f.operator == "in":
            values = [v.strip() for v in f.value.split(",")]
            df = df.filter(col.isin(values))
        elif f.operator == "not_in":
            values = [v.strip() for v in f.value.split(",")]
            df = df.filter(~col.isin(values))
    return df


def _apply_aggregations(df: DataFrame, plan: ActionPlan) -> DataFrame:
    aggs: List = []
    for a in plan.aggregations or []:
        func = a.function.lower()
        if func == "sum":
            expr = F.sum(F.col(a.column))
        elif func == "avg":
            expr = F.avg(F.col(a.column))
        elif func == "min":
            expr = F.min(F.col(a.column))
        elif func == "max":
            expr = F.max(F.col(a.column))
        elif func == "count":
            expr = F.count(F.col(a.column))
        else:
            raise ValueError(f"Unsupported aggregation: {func}")
        if a.alias:
            expr = expr.alias(a.alias)
        aggs.append(expr)
    if plan.group_by:
        return df.groupBy(*plan.group_by).agg(*aggs)
    return df.agg(*aggs)


def execute_plan(input_uri: str, plan: ActionPlan, settings: Settings, metadata: dict) -> DataFrame:
    spark = get_spark(settings)
    logger.info("Reading input from %s", input_uri)
    df = _read_input(spark, input_uri)

    if plan.filters:
        df = _apply_filters(df, plan.filters)

    if plan.aggregations:
        df = _apply_aggregations(df, plan)

    if plan.select:
        df = df.select(*plan.select)

    if plan.order_by:
        order_cols = [
            F.col(o.column).desc() if o.direction == "desc" else F.col(o.column).asc()
            for o in plan.order_by
        ]
        df = df.orderBy(*order_cols)

    if plan.limit:
        df = df.limit(plan.limit)

    return df

