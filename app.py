import os
import logging
import traceback

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, BaseSettings

from t5qg import T5

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


# Initialization
MODEL = os.getenv('MODEL', 'asahi417/question-generation-squad-t5-small')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
MAX_LENGTH_OUTPUT = int(os.getenv('MAX_LENGTH_OUTPUT', 32))

qg_model = T5(MODEL, MAX_LENGTH, MAX_LENGTH_OUTPUT)


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
        "model": MODEL,
        "max_length": MAX_LENGTH,
        "max_length_output": MAX_LENGTH_OUTPUT,
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


