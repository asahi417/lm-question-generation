""" API with huggingface inference API

Language-wise best model
- answer specified
- answer agnostic
"""
import os
import logging
import random
import traceback
from typing import List

from transformers import AutoConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lmqg.inference_api import generate_qa
from lmqg.spacy_module import SpacyPipeline


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

API_TOKEN = os.getenv("API_TOKEN")
KEYWORD_EXTRACTOR = os.getenv('KEYWORD_EXTRACTOR', 'positionrank')
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
DEFAULT_MODELS_E2E = {
    "en": "lmqg/t5-base-squad-qag",
    "ja": "lmqg/mt5-base-jaquad-qag",
    "de": "lmqg/mt5-small-dequad-qg",
    "es": "lmqg/mt5-base-esquad-qag",
    "it": "lmqg/mt5-base-itquad-qag",
    "ko": "lmqg/lmqg/mbart-large-cc25-koquad-qag",
    "fr": "lmqg/mt5-base-frquad-qag",
    "ru": "lmqg/mt5-base-ruquad-qg-ae"
}
DEFAULT_MODELS_PIPELINE = {
    "en": ["lmqg/bart-large-squad-qg", "lmqg/bart-large-squad-ae"],
    "ja": ["lmqg/mt5-small-jaquad-qg-ae", "lmqg/mt5-small-jaquad-qg-ae"],
    "de": ["lmqg/mt5-small-dequad-qg", "lmqg/mt5-small-dequad-ae"],
    "es": ["lmqg/mt5-base-esquad-qg", "lmqg/mt5-base-esquad-ae"],
    "it": ["lmqg/mt5-base-itquad-qg-ae", "lmqg/mt5-base-itquad-qg-ae"],
    "ko": ["lmqg/mbart-large-cc25-koquad-qg", "lmqg/mbart-large-cc25-koquad-ae"],
    "fr": ["lmqg/mt5-small-frquad-qg", "lmqg/mt5-small-frquad-ae"],
    "ru": ["lmqg/mt5-base-ruquad-qg-ae", "lmqg/mt5-base-ruquad-qg-ae"]
}
SUPPORT_LANGUAGE = list(LANGUAGE_MAP.values())
QG_PRETTY_NAME_ = {f'mT5 SMALL ({i.upper()})': f'lmqg/mt5-small-{i}quad' for i in SUPPORT_LANGUAGE if i != 'en'}
QG_PRETTY_NAME.update({
    "T5 SMALL": "lmqg/t5-small-squad", "T5 BASE": "lmqg/t5-base-squad", "T5 LARGE": "lmqg/t5-large-squad",
    "BART BASE": "lmqg/bart-base-squad", "BART LARGE": "lmqg/bart-large-squad"})
QG_PRETTY_NAME.update({f'mT5 BASE ({i.upper()})': f'lmqg/mt5-base-{i}quad' for i in SUPPORT_LANGUAGE if i != 'en'})
QG_PRETTY_NAME.update({f'mBART LARGE ({i.upper()})': f'lmqg/mbart-large-cc25-{i}quad' for i in SUPPORT_LANGUAGE if i != 'en'})

PREFIX_INFO_QAG = {k: AutoConfig.from_pretrained(f"{v}-qag").add_prefix for k, v in QG_PRETTY_NAME.items()}
PREFIX_INFO_QG_AE = {k: AutoConfig.from_pretrained(f"{v}-qg-ae").add_prefix for k, v in QG_PRETTY_NAME.items()}
PREFIX_INFO_QG = {k: AutoConfig.from_pretrained(f"{v}-qg").add_prefix for k, v in QG_PRETTY_NAME.items()}
PREFIX_INFO_AE = {k: AutoConfig.from_pretrained(f"{v}-ae").add_prefix for k, v in QG_PRETTY_NAME.items()}


# Run app
class ModelInput(BaseModel):
    input_text: str
    language: str = 'en'
    answer_model: str = None
    qg_model: str = 'lmqg/t5-small-squad-qag'
    highlight: str or List = None
    num_beams: int = 4
    num_questions: int = 5
    use_gpu: bool = False
    do_sample: bool = True
    top_p: float = 0.9
    max_length: int = 64


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Endpoint
@app.get("/")
def read_root():
    return {"What's this?": "Automatic question & answer generation application."}


@app.get("/info")
async def info():
    return {"QG models": QG_PRETTY_NAME}


@app.post("/question_generation")
async def process(model_input: ModelInput):
    if len(model_input.input_text) == 0:
        raise HTTPException(status_code=404, detail='Input text is empty string.')
    try:

        if model_input.highlight is None:
            prefix_qg = QG_PREFIX_INFO_QAG[model_input.qg_model] if model_input.qg_model in QG_PRETTY_NAME else None
            qg_model = f"{QG_PRETTY_NAME[model_input.qg_model]}-qag" if model_input.qg_model in QG_PRETTY_NAME else model_input.qg_model
        else:
            prefix_qg = QG_PREFIX_INFO_QG_AE[model_input.qg_model] if model_input.qg_model in QG_PRETTY_NAME else None
            qg_model = f"{QG_PRETTY_NAME[model_input.qg_model]}-qg-ae" if model_input.qg_model in QG_PRETTY_NAME else model_input.qg_model
            prefix_qg = QG_PREFIX_INFO_QG_AE[model_input.qg_model] if model_input.qg_model in QG_PRETTY_NAME else None
            prefix_ae = AE_PREFIX_INFO[model_input.answer_model]
            model_input.answer_model = AE_PRETTY_NAME[model_input.answer_model]
        if model_input.language in LANGUAGE_MAP:
            model_input.language = LANGUAGE_MAP[model_input.language]
        qa_list = generate_qa(
            api_token=API_TOKEN,
            input_text=model_input.input_text,
            model_qg=model_input.qg_model,
            spacy=None,
            model_ae=model_input.answer_model,
            input_answer=model_input.highlight,
            top_p=model_input.top_p,
            use_gpu=model_input.use_gpu,
            do_sample=model_input.do_sample,
            max_length=model_input.max_length,
            num_beams=model_input.num_beams,
            # is_qag: bool = None,
            add_prefix_qg=prefix_qg,
            add_prefix_answer=prefix_answer
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
