from __future__ import annotations

import logging

from llm_spark_pipeline.config import Settings
from llm_spark_pipeline.io.output import write_output
from llm_spark_pipeline.llm.extractor import extract_action_plan
from llm_spark_pipeline.spark.executor import execute_plan
from llm_spark_pipeline.validation.validator import validate_plan

logger = logging.getLogger(__name__)


def run_pipeline(
    question: str,
    metadata: dict,
    input_uri: str,
    output_uri: str,
    settings: Settings,
) -> str:
    last_error = None
    plan = None

    for attempt in range(settings.max_retries + 1):
        plan = extract_action_plan(
            question=question,
            metadata=metadata,
            settings=settings,
            previous_error=last_error,
        )
        errors = validate_plan(plan, metadata)
        if not errors:
            last_error = None
            break
        last_error = "; ".join(errors)
        logger.warning("Validation failed (attempt %d): %s", attempt + 1, last_error)

    if not plan:
        raise ValueError("No plan could be generated")
    if last_error:
        raise ValueError(f"Validation failed after retries: {last_error}")

    df = execute_plan(input_uri, plan, settings, metadata)
    output_path = write_output(df, output_uri, settings)
    return output_path
