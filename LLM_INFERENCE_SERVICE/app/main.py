from fastapi import FastAPI, HTTPException
from app.model import ModelService
from app.schema import GeneralRequest, GenerateResponse
from contextlib import asynccontextmanager


hf_model_name = "distilgpt2"
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService(hf_model_name, device)
    yield

app = FastAPI(lifespan=lifespan)

@app.post('/generate', response_model=GenerateResponse)
def generate_text(request: GeneralRequest):
    try:
        ouput = app.state.model_service.generate(

            request.prompt,
            request.max_tokens,
            request.temperature
        )
        return GenerateResponse(output=ouput)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

