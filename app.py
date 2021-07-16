import logging
import traceback

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, BaseSettings

from t5qg import T5

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


# Initialization
class Settings(BaseSettings):
    model: str = 'asahi417/question-generation-squad-t5-small'
    max_length: int = 512
    max_length_output: int = 32


settings = Settings()
qg_model = T5(settings.model, settings.max_length, settings.max_length_output)


# Run app
class ModelInput(BaseModel):
    input_text: str
    highlight: Optional[str] = None
    task: str = 'qg'
    num_beam: int = 4


app = FastAPI()


# Endpoint
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
    try:
        out = qg_model.get_prediction([model_input.input_text],
                                      [model_input.highlight],
                                      num_beams=model_input.num_beam,
                                      task_prefix=model_input.task)
        return out
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


