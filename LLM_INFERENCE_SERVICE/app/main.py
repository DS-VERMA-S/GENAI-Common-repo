import logging
import os
import time
from contextlib import asynccontextmanager
from functools import wraps

from fastapi import FastAPI, HTTPException

from app.model import ModelService
from app.schema import GenerateRequest, GenerateResponse


hf_model_name = "Qwen/Qwen3-0.6B"
device = "cpu"

logger = logging.getLogger("llm_inference_service")
if not logger.handlers:
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "llm_inference_service.log")
    log_format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def track_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("%s took %.2f ms", func.__name__, elapsed_ms)

    return wrapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService(hf_model_name, device)
    yield

app = FastAPI(lifespan=lifespan)



@app.post('/generate', response_model=GenerateResponse)
@track_time
def generate_text(request: GenerateRequest):
    try:
        output = app.state.model_service.generate(

            request.prompt,
            request.max_tokens,
            request.temperature
        )
        logger.info(
            "Generated response (prompt_len=%d, max_tokens=%d, temperature=%.3f)",
            len(request.prompt),
            request.max_tokens,
            request.temperature,
        )
        return GenerateResponse(output=output)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))
