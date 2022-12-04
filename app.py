""" API with huggingface inference API """
import os
import logging
import random
import traceback
from typing import List

from transformers import AutoConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lmqg import TransformersQGInferenceAPI

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

API_TOKEN = os.getenv("API_TOKEN")
KEYWORD_EXTRACTOR = os.getenv('KEYWORD_EXTRACTOR', 'positionrank')

QG_PRETTY_NAME = {
    'T5 SMALL': 'lmqg/t5-small-squad-qg-ae',
    'T5 BASE': 'lmqg/t5-base-squad-qg-ae',
    'T5 LARGE': 'lmqg/t5-large-squad-qg-ae',
    'BART BASE': 'lmqg/bart-base-squad-qg',
    'BART LARGE': 'lmqg/bart-large-squad-qg',
    'mT5 SMALL (JA)': 'lmqg/mt5-small-jaquad-qg-ae',
    'mT5 SMALL (DE)': 'lmqg/mt5-small-dequad-qg-ae',
    'mT5 SMALL (ES)': 'lmqg/mt5-small-esquad-qg-ae',
    'mT5 SMALL (KO)': 'lmqg/mt5-small-koquad-qg-ae',
    'mT5 SMALL (RU)': 'lmqg/mt5-small-ruquad-qg-ae',
    'mT5 SMALL (IT)': 'lmqg/mt5-small-itquad-qg-ae',
    'mT5 SMALL (FR)': 'lmqg/mt5-small-frquad-qg-ae',
    'mT5 BASE (JA)': 'lmqg/mt5-base-jaquad-qg',
    'mT5 BASE (DE)': 'lmqg/mt5-base-dequad-qg',
    'mT5 BASE (ES)': 'lmqg/mt5-base-esquad-qg',
    'mT5 BASE (KO)': 'lmqg/mt5-base-koquad-qg',
    'mT5 BASE (RU)': 'lmqg/mt5-base-ruquad-qg',
    'mT5 BASE (IT)': 'lmqg/mt5-base-itquad-qg',
    'mT5 BASE (FR)': 'lmqg/mt5-base-frquad-qg',
    'mBART LARGE (JA)': 'lmqg/mbart-large-cc25-jaquad-qg',
    'mBART LARGE (DE)': 'lmqg/mbart-large-cc25-dequad-qg',
    'mBART LARGE (ES)': 'lmqg/mbart-large-cc25-esquad-qg',
    'mBART LARGE (KO)': 'lmqg/mbart-large-cc25-koquad-qg',
    'mBART LARGE (RU)': 'lmqg/mbart-large-cc25-ruquad-qg',
    'mBART LARGE (IT)': 'lmqg/mbart-large-cc25-itquad-qg',
    'mBART LARGE (FR)': 'lmqg/mbart-large-cc25-frquad-qg'
}
QG_PREFIX_INFO = {k: AutoConfig.from_pretrained(v).add_prefix for k, v in QG_PRETTY_NAME.items()}
AE_PRETTY_NAME = {
    'T5 SMALL': 'lmqg/t5-small-squad-qg-ae',
    'T5 BASE': 'lmqg/t5-base-squad-qg-ae',
    'T5 LARGE': 'lmqg/t5-large-squad-qg-ae',
    'mT5 SMALL (JA)': 'lmqg/mt5-small-jaquad-qg-ae',
    'mT5 SMALL (DE)': 'lmqg/mt5-small-dequad-qg-ae',
    'mT5 SMALL (ES)': 'lmqg/mt5-small-esquad-qg-ae',
    'mT5 SMALL (KO)': 'lmqg/mt5-small-koquad-qg-ae',
    'mT5 SMALL (RU)': 'lmqg/mt5-small-ruquad-qg-ae',
    'mT5 SMALL (IT)': 'lmqg/mt5-small-itquad-qg-ae',
    'mT5 SMALL (FR)': 'lmqg/mt5-small-frquad-qg-ae',
    'Keyword': None
}
AE_PREFIX_INFO = {k: AutoConfig.from_pretrained(v).add_prefix if v is not None else None
                  for k, v in AE_PRETTY_NAME.items()}
LANGUAGE_MAP = {
    "Japanese": "ja",
    "English": "en",
    'German': "de",
    'Spanish': 'es',
    'Italian': 'it',
    'Korean': 'ko',
    'Russian': "ru",
    'French': "fr"
}


# Run app
class ModelInput(BaseModel):
    input_text: str
    language: str = 'en'
    answer_model: str = 'lmqg/t5-small-squad-qg-ae'  # 'keyword_extraction'
    qg_model: str = 'lmqg/t5-small-squad-qg'
    highlight: str or List = None
    num_beams: int = 4
    num_questions: int = 5
    use_gpu: bool = False
    do_sample: bool = True
    top_p: float = 0.9
    max_length: int = 64


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Endpoint
@app.get("/")
def read_root():
    return {"What's this?": "Automatic question & answer generation application."}


@app.get("/info")
async def info():
    return {"keyword_extractor": KEYWORD_EXTRACTOR, "QG models": QG_PRETTY_NAME, "Answer models": AE_PRETTY_NAME}


@app.post("/question_generation")
async def process(model_input: ModelInput):
    if len(model_input.input_text) == 0:
        raise HTTPException(status_code=404, detail='Input text is empty string.')
    try:
        prefix_qg = None
        prefix_answer = None
        if model_input.qg_model in QG_PRETTY_NAME:
            prefix_qg = QG_PREFIX_INFO[model_input.qg_model]
            model_input.qg_model = QG_PRETTY_NAME[model_input.qg_model]
        if model_input.answer_model in AE_PRETTY_NAME:
            prefix_answer = AE_PREFIX_INFO[model_input.answer_model]
            model_input.answer_model = AE_PRETTY_NAME[model_input.answer_model]
        if model_input.language in LANGUAGE_MAP:
            model_input.language = LANGUAGE_MAP[model_input.language]
        qg_model = TransformersQGInferenceAPI(
            model=model_input.qg_model,
            answer_model=model_input.answer_model,
            api_token=API_TOKEN,
            language=model_input.language,
            add_prefix_qg=prefix_qg,
            add_prefix_answer=prefix_answer,
            keyword_extraction_model=KEYWORD_EXTRACTOR)
        qa_list = qg_model.generate_qa(
            model_input.input_text,
            input_answer=model_input.highlight,
            num_beams=model_input.num_beams,
            num_questions=model_input.num_questions,
            top_p=model_input.top_p,
            use_gpu=model_input.use_gpu,
            do_sample=model_input.do_sample,
            max_length=model_input.max_length
        )
        return {'qa': qa_list}
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


@app.post("/question_generation_dummy")
async def process(model_input: ModelInput):
    i = random.randint(0, 2)
    target = [{'question': "Who founded Nintendo Karuta?", 'answer': "Fusajiro Yamauchi"},
              {'question': "When did Nintendo distribute its first video game console, the Color TV-Game?", 'answer': "1977"},
              {'question': "When did Nintendo release Super Mario Bros?", 'answer': "1985"}]
    return {"qa": target[:i]}
