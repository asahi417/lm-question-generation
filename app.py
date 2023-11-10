""" API with huggingface inference API """
import os
import logging
import random
import traceback
from difflib import SequenceMatcher
from typing import List
from itertools import chain

from transformers import AutoConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lmqg.inference_api import generate_qa
from lmqg.spacy_module import SpacyPipeline

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

API_TOKEN = os.getenv("API_TOKEN")

##########
# CONFIG #
##########
LANGUAGE_MAP = {
    "English": "en",
    "Japanese": "ja",
    'German': "de",
    'Spanish': 'es',
    'Italian': 'it',
    'Korean': 'ko',
    'Russian': "ru",
    'French': "fr",
    'Chinese': "zh"
}
# Default models for each qag type
DEFAULT_MODELS_E2E = {
    "en": ["lmqg/t5-small-squad-qag", None],
    "fr": ["lmqg/mt5-small-frquad-qag-trimmed-50000", None],
    "es": ["lmqg/mt5-small-esquad-qag-trimmed-50000", None],
    "it": ["lmqg/mt5-small-itquad-qag-trimmed-50000", None],
    "ja": ["lmqg/mt5-small-jaquad-qag-trimmed-50000", None],
    "de": ["lmqg/mt5-small-dequad-qag-trimmed-50000", None],
    "ko": ["lmqg/mt5-small-koquad-qag-trimmed-50000", None],
    "ru": ["lmqg/mt5-small-ruquad-qag-trimmed-50000", None],
    "zh": ["lmqg/mt5-small-zhquad-qag-trimmed-50000", None]
}
DEFAULT_MODELS_MULTITASK = {
    "en": ["lmqg/t5-small-squad-qg-ae", "lmqg/t5-small-squad-qg-ae"],
    "fr": ["lmqg/mt5-small-frquad-qg-ae-trimmed-50000", "lmqg/mt5-small-frquad-qg-ae-trimmed-50000"],
    "es": ["lmqg/mt5-small-esquad-qg-ae-trimmed-50000", "lmqg/mt5-small-esquad-qg-ae-trimmed-50000"],
    "it": ["lmqg/mt5-small-itquad-qg-ae-trimmed-50000", "lmqg/mt5-small-itquad-qg-ae-trimmed-50000"],
    "ja": ["lmqg/mt5-small-jaquad-qg-ae-trimmed-50000", "lmqg/mt5-small-jaquad-qg-ae-trimmed-50000"],
    "de": ["lmqg/mt5-small-dequad-qg-ae-trimmed-50000", "lmqg/mt5-small-dequad-qg-ae-trimmed-50000"],
    "ko": ["lmqg/mt5-small-koquad-qg-ae-trimmed-50000", "lmqg/mt5-small-koquad-qg-ae-trimmed-50000"],
    "ru": ["lmqg/mt5-small-ruquad-qg-ae-trimmed-50000", "lmqg/mt5-small-ruquad-qg-ae-trimmed-50000"],
    "zh": ["lmqg/mt5-small-zhquad-qg-ae-trimmed-50000", "lmqg/mt5-small-zhquad-qg-ae-trimmed-50000"]
}
DEFAULT_MODELS_PIPELINE = {
    "en": ["lmqg/t5-small-squad-qg", "lmqg/t5-small-squad-ae"],
    "fr": ["lmqg/mt5-small-frquad-qg-trimmed-50000", "lmqg/mt5-small-frquad-ae-trimmed-50000"],
    "es": ["lmqg/mt5-small-esquad-qg-trimmed-50000", "lmqg/mt5-small-esquad-ae-trimmed-50000"],
    "it": ["lmqg/mt5-small-itquad-qg-trimmed-50000", "lmqg/mt5-small-itquad-ae-trimmed-50000"],
    "ja": ["lmqg/mt5-small-jaquad-qg-trimmed-50000", "lmqg/mt5-small-jaquad-ae-trimmed-50000"],
    "de": ["lmqg/mt5-small-dequad-qg-trimmed-50000", "lmqg/mt5-small-dequad-ae-trimmed-50000"],
    "ko": ["lmqg/mt5-small-koquad-qg-trimmed-50000", "lmqg/mt5-small-koquad-ae-trimmed-50000"],
    "ru": ["lmqg/mt5-small-ruquad-qg-trimmed-50000", "lmqg/mt5-small-ruquad-qg-trimmed-50000"],
    "zh": ["lmqg/mt5-small-zhquad-qg-trimmed-50000", "lmqg/mt5-small-zhquad-qg-trimmed-50000"]
}
DEFAULT_MODELS = {
    "End2End": DEFAULT_MODELS_E2E,
    "Multitask": DEFAULT_MODELS_PIPELINE,
    "Pipeline": DEFAULT_MODELS_PIPELINE,
}
SPACY_PIPELINE = {i: SpacyPipeline(i) for i in LANGUAGE_MAP.values()}  # spacy for sentence splitting
PRETTY_NAME = {
    "End2End": {
        "T5 SMALL": ["lmqg/t5-small-squad-qag", None],
        "T5 BASE": ["lmqg/t5-base-squad-qag", None],
        "T5 LARGE": ["lmqg/t5-large-squad-qag", None],
        "Flan-T5 SMALL": ["lmqg/flan-t5-small-squad-qag", None],
        "Flan-T5 BASE": ["lmqg/flan-t5-base-squad-qag", None],
        "Flan-T5 LARGE": ["lmqg/flan-t5-large-squad-qag", None]
    },
    "Multitask": {
        "T5 SMALL": ["lmqg/t5-small-squad-qg-ae", "lmqg/t5-small-squad-qg-ae"],
        "T5 BASE": ["lmqg/t5-base-squad-qg-ae", "lmqg/t5-base-squad-qg-ae"],
        "T5 LARGE": ["lmqg/t5-large-squad-qg-ae", "lmqg/t5-large-squad-qg-ae"],
        "Flan-T5 SMALL": ["lmqg/flan-t5-small-squad-qg-ae", "lmqg/flan-t5-small-squad-qg-ae"],
        "Flan-T5 BASE": ["lmqg/flan-t5-base-squad-qg-ae", "lmqg/flan-t5-base-squad-qg-ae"],
        "Flan-T5 LARGE": ["lmqg/flan-t5-large-squad-qg-ae", "lmqg/flan-t5-large-squad-qg-ae"],
    },
    "Pipeline": {
        "T5 SMALL": ["lmqg/t5-small-squad-qg", "lmqg/t5-small-squad-ae"],
        "T5 BASE": ["lmqg/t5-base-squad-qg", "lmqg/t5-base-squad-ae"],
        "T5 LARGE": ["lmqg/t5-large-squad-qg", "lmqg/t5-large-squad-ae"],
        "Flan-T5 SMALL": ["lmqg/flan-t5-small-squad-qg", "lmqg/flan-t5-small-squad-ae"],
        "Flan-T5 BASE": ["lmqg/flan-t5-base-squad-qg", "lmqg/flan-t5-base-squad-ae"],
        "Flan-T5 LARGE": ["lmqg/flan-t5-large-squad-qg", "lmqg/flan-t5-large-squad-ae"],
    }
}
PRETTY_NAME["End2End"].update({
    f'mT5 SMALL ({i.upper()})': [f'lmqg/mt5-small-{i}quad-qag-trimmed-50000', None]
    for i in LANGUAGE_MAP.values() if i != 'en'
})
PRETTY_NAME["End2End"].update({
    f'mT5 BASE ({i.upper()})': [f'lmqg/mt5-base-{i}quad-qag-trimmed-50000', None]
    for i in LANGUAGE_MAP.values() if i != 'en'
})
PRETTY_NAME["Multitask"].update({
    f'mT5 SMALL ({i.upper()})': [
        f'lmqg/mt5-small-{i}quad-qg-ae-trimmed-50000',
        f'lmqg/mt5-small-{i}quad-qg-ae-trimmed-50000'
    ] for i in LANGUAGE_MAP.values() if i != 'en'
})
PRETTY_NAME["Multitask"].update({
    f'mT5 BASE ({i.upper()})': [
        f'lmqg/mt5-base-{i}quad-qg-ae-trimmed-50000',
        f'lmqg/mt5-base-{i}quad-qg-ae-trimmed-50000'
    ] for i in LANGUAGE_MAP.values() if i != 'en'
})
PRETTY_NAME["Pipeline"].update({
    f'mT5 SMALL ({i.upper()})': [
        f'lmqg/mt5-small-{i}quad-qg-trimmed-50000',
        f'lmqg/mt5-small-{i}quad-ae-trimmed-50000'
    ] for i in LANGUAGE_MAP.values() if i != 'en'
})
PRETTY_NAME["Pipeline"].update({
    f'mT5 BASE ({i.upper()})': [
        f'lmqg/mt5-base-{i}quad-qg-trimmed-50000',
        f'lmqg/mt5-base-{i}quad-ae-trimmed-50000'
    ] for i in LANGUAGE_MAP.values() if i != 'en'
})
# Prefix information for each model
PREFIX_INFO_QAG = {
    v: AutoConfig.from_pretrained(v).add_prefix
    for v in chain(*chain(*[v.values() for v in DEFAULT_MODELS.values()])) if v is not None
}


########
# MAIN #
########
# App input
class ModelInput(BaseModel):
    input_text: str
    language: str = 'en'
    qag_type: str = None  # End2End/Pipeline
    model: str = None
    highlight: str or List = None  # answer
    num_beams: int = 4
    use_gpu: bool = False
    do_sample: bool = True
    top_p: float = 0.9
    max_length: int = 64
    split: str = 'paragraph'


# Run app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def validate_default(value, default=None):
    if value is not None and type(value) is not str:
        return value
    if value is None or value.lower() in ['default', '']:
        return default
    return value


############
# ENDPOINT #
############
# General info
@app.get("/")
def read_root():
    return {"About": "Automatic question & answer generation application. Send inquiry to https://asahiushio.com/."}


# Main endpoint
@app.post("/question_generation")
async def process(model_input: ModelInput):
    if len(model_input.input_text) == 0:
        raise HTTPException(status_code=404, detail='Input text is empty string.')
    try:
        # language validation
        if model_input.language in LANGUAGE_MAP:
            model_input.language = LANGUAGE_MAP[model_input.language]
        # qag_type validation
        elif model_input.qag_type == 'End2End' and model_input.highlight is not None:
            raise ValueError("End2End qag_type does not support answer-given question generation")

        model_input.highlight = validate_default(model_input.highlight, default=None)
        model_input.qag_type = validate_default(
            model_input.qag_type,
            default='Multitask'
        )
        model_input.model = validate_default(model_input.model, default=None)

        # model validation
        if model_input.model is None:
            model_qg, model_ae = DEFAULT_MODELS[model_input.qag_type][model_input.language]
        else:
            model_qg, model_ae = PRETTY_NAME[model_input.qag_type][model_input.model]
        qa_list = generate_qa(
            api_token=API_TOKEN,
            input_text=model_input.input_text,
            model_qg=model_qg,
            model_ae=model_ae,
            spacy=SPACY_PIPELINE[model_input.language],
            input_answer=model_input.highlight,
            top_p=model_input.top_p,
            use_gpu=model_input.use_gpu,
            do_sample=model_input.do_sample,
            max_length=model_input.max_length,
            num_beams=model_input.num_beams,
            is_qag=model_input.qag_type == 'End2End',
            add_prefix_qg=PREFIX_INFO_QAG[model_qg] if model_qg in PREFIX_INFO_QAG else None,
            add_prefix_answer=PREFIX_INFO_QAG[model_ae] if model_ae is not None and model_ae in PREFIX_INFO_QAG else None,
            split_level=model_input.split.lower()
        )
        for qa in qa_list:
            q = qa['question']
            c = model_input.input_text
            match = SequenceMatcher(None, c, q).find_longest_match(0, len(c), 0, len(q))
            qa['score'] = 1 - match.size / len(q)
        qa_list = sorted(qa_list, key=lambda x: x['score'], reverse=True)
        return {'qa': qa_list}
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


# Dummy endpoint
@app.post("/question_generation_dummy")
async def process(model_input: ModelInput):
    i = random.randint(0, 2)
    target = [{'question': "Who founded Nintendo Karuta?", 'answer': "Fusajiro Yamauchi", "score": 0.5},
              {'question': "When did Nintendo distribute its first video game console, the Color TV-Game?", 'answer': "1977", "score": 0.3},
              {'question': "When did Nintendo release Super Mario Bros?", 'answer': "1985", "score": 0.1}]
    return {"qa": target[:i]}
