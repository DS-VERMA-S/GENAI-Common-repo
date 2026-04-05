from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from llm_spark_pipeline.config import load_settings
from llm_spark_pipeline.logging import setup_logging
from llm_spark_pipeline.pipeline import run_pipeline

app = typer.Typer(add_completion=False, help="LLM + Spark pipeline CLI")
console = Console()


@app.command()
def main(
    question: Optional[str] = typer.Option(None, "--question", help="User question"),
    question_file: Optional[Path] = typer.Option(
        None, "--question-file", exists=True, help="Path to question text file"
    ),
    metadata: Optional[str] = typer.Option(
        None, "--metadata", help="Metadata JSON string"
    ),
    metadata_file: Optional[Path] = typer.Option(
        None, "--metadata-file", exists=True, help="Path to metadata JSON file"
    ),
    input_uri: str = typer.Option(..., "--input-uri", help="Input dataset URI (S3)"),
    output_uri: str = typer.Option(..., "--output-uri", help="Output URI (S3)"),
    output_format: Optional[str] = typer.Option(
        None, "--output-format", help="Output format (excel or parquet)"
    ),
    max_retries: Optional[int] = typer.Option(
        None, "--max-retries", help="Max LLM retries on validation failure"
    ),
):
    settings = load_settings()
    setup_logging(settings.log_level)

    if not question and not question_file:
        raise typer.BadParameter("Provide --question or --question-file")
    if not metadata and not metadata_file:
        raise typer.BadParameter("Provide --metadata or --metadata-file")

    if question_file:
        question_text = question_file.read_text(encoding="utf-8").strip()
    else:
        question_text = question or ""

    if metadata_file:
        metadata_obj = json.loads(metadata_file.read_text(encoding="utf-8"))
    else:
        metadata_obj = json.loads(metadata or "{}")

    if output_format:
        settings.output_format = output_format
    if max_retries is not None:
        settings.max_retries = max_retries

    result = run_pipeline(
        question=question_text,
        metadata=metadata_obj,
        input_uri=input_uri,
        output_uri=output_uri,
        settings=settings,
    )

    console.print(f"[green]Output written to:[/green] {result}")


if __name__ == "__main__":
    app()

