from __future__ import annotations

import logging
import os
import tempfile

from pyspark.sql import DataFrame

from llm_spark_pipeline.config import Settings
from llm_spark_pipeline.io.s3 import upload_file

logger = logging.getLogger(__name__)


def write_output(df: DataFrame, output_uri: str, settings: Settings) -> str:
    output_format = settings.output_format.lower()
    if output_format == "parquet":
        df.write.mode("overwrite").parquet(output_uri)
        return output_uri

    if output_format != "excel":
        raise ValueError("Unsupported output format")

    row_limit = settings.excel_row_limit
    pandas_df = df.limit(row_limit).toPandas()

    fd, local_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    pandas_df.to_excel(local_path, index=False)

    logger.info("Uploading excel output to %s", output_uri)
    return upload_file(local_path, output_uri)

