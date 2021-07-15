from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings

from t5qg import T5


# Initialization
class Settings(BaseSettings):
    model: str
    max_length: int = 512
    max_length_output: int = 32


settings = Settings()
qg_model = T5(settings.model, settings.max_length, settings.max_length_output)


# Run app
class ModelInput(BaseModel):
    input_text: str
    highlight: Optional[str] = None
    question: Optional[str] = None
    task: Optional[str] = 'qg'


app = FastAPI()


# Endpoint
# qg_model.get_prediction()
@app.get("/")
def read_root():
    return {"What's this?": "Awesome question generation web App ever!"}


@app.get("/info")
async def info():
    return {
        "model": settings.model,
        "max_length": settings.max_length,
        "max_length_output": settings.max_length_output,
    }


@app.post("/question_generation")
async def process(model_input: ModelInput):
    return model_input


@app.post("/answer_extraction")
async def process(model_input: ModelInput):
    return model_input


@app.post("/question_answering")
async def process(model_input: ModelInput):
    return model_input

