from __future__ import annotations

import json
import logging
from typing import Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from llm_spark_pipeline.config import Settings
from llm_spark_pipeline.llm.prompt_templates import BASE_PROMPT, RETRY_PROMPT
from llm_spark_pipeline.validation.schema import ActionPlan

logger = logging.getLogger(__name__)


def _llm_client(settings: Settings) -> AzureChatOpenAI:
    if not settings.azure_openai_endpoint or not settings.azure_openai_key:
        raise ValueError("Azure OpenAI credentials are missing")
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )


def extract_action_plan(
    question: str,
    metadata: dict,
    settings: Settings,
    previous_error: Optional[str] = None,
) -> ActionPlan:
    parser = PydanticOutputParser(pydantic_object=ActionPlan)
    prompt_text = RETRY_PROMPT if previous_error else BASE_PROMPT
    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["question", "metadata", "errors"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    payload = {
        "question": question,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "errors": previous_error or "",
    }

    llm = _llm_client(settings)
    result = llm.invoke(prompt.format(**payload))
    text = result.content if hasattr(result, "content") else str(result)

    logger.info("LLM raw response: %s", text)
    try:
        return parser.parse(text)
    except OutputParserException:
        cleaned = _strip_code_fence(text)
        return parser.parse(cleaned)


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()
