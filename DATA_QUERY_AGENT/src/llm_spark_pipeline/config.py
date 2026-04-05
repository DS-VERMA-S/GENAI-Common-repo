from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    azure_openai_endpoint: str = Field("", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field("", alias="AZURE_OPENAI_KEY")
    azure_openai_deployment: str = Field("", alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field("2024-06-01", alias="AZURE_OPENAI_API_VERSION")

    aws_access_key_id: str = Field("", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field("", alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field("", alias="AWS_DEFAULT_REGION")

    spark_master: str = Field("local[*]", alias="SPARK_MASTER")
    spark_app_name: str = Field("llm-spark-pipeline", alias="SPARK_APP_NAME")

    output_format: str = Field("excel", alias="OUTPUT_FORMAT")
    excel_row_limit: int = Field(100000, alias="EXCEL_ROW_LIMIT")
    max_retries: int = Field(2, alias="MAX_RETRIES")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


def load_settings() -> Settings:
    return Settings()
