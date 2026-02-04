from pydantic import BaseModel, Field

class GeneralRequest(BaseModel) : 

    prompt: str = Field(..., min_length=1, max_length=500)
    max_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.2, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    output: str
