from pydantic import BaseModel, Field

from pydantic import field_validator

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.2, ge=0.0, le=1.0)

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str):
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v



class GenerateResponse(BaseModel):
    output: str
