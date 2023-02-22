""" Language model for Huggingface inference API. """
import requests
import json
import logging
import re
from typing import List

import transformers
from .exceptions import APIError, HighlightNotFoundError, AnswerNotFoundError
from .spacy_module import SpacyPipeline
from .language_model import clean, TASK_PREFIX, ADDITIONAL_SP_TOKENS


def call_api(input_text,
             api_token,
             model,
             max_length: int = 64,
             do_sample: bool = True,
             num_beams: int = 1,
             top_p: float = 0.9,
             use_gpu: bool = False):
    data = json.dumps({
        "inputs": input_text,
        "parameters": {"do_sample": do_sample, "num_beams": num_beams, "max_length": max_length, "top_p": top_p},
        "options": {"wait_for_model": True, "use_gpu": use_gpu}
    })
    response = requests.request("POST", f"https://api-inference.huggingface.co/models/{model}",
                                headers={"Authorization": f"Bearer {api_token}"}, data=data)
    data = response.json()
    if type(data) is dict and 'error' in data.keys():
        raise APIError(data)
    return data


def highlight_sentence(input_text: str, highlight_span: str, prefix: str = None):
    position = input_text.find(highlight_span)
    if position == -1:
        HighlightNotFoundError(highlight_span, input_text)
    input_text = f"{input_text[:position]}{ADDITIONAL_SP_TOKENS['hl']} " \
                 f"{highlight_span} {ADDITIONAL_SP_TOKENS['hl']}{input_text[position + len(highlight_span):]}"
    if prefix is not None:
        input_text = f"{prefix}: {input_text}"
    return input_text


def generate_qa(
        api_token: str,
        input_text: str,
        model_qg: str,
        spacy: SpacyPipeline = None,
        model_ae: str = None,
        input_answer: List or str = None,
        max_length: int = 256,
        do_sample: bool = True,
        num_beams: int = 1,
        top_p: float = 0.9,
        use_gpu: bool = False,
        is_qag: bool = None,
        add_prefix_qg: bool = None,
        add_prefix_answer: bool = None,
        splitting_symbol: str = '|',
        question_prefix: str = "question: ",
        answer_prefix: str = ", answer: ",
        split_level: str = "paragraph"):

    logging.info('initialize TransformersQGInferenceAPI')
    is_qag = model_qg.endswith('qag') if is_qag is None else is_qag
    add_prefix_qg = transformers.AutoConfig.from_pretrained(model_qg).add_prefix if add_prefix_qg is None else add_prefix_qg

    qa = []
    qa_hash = []
    if is_qag:
        assert input_answer is None
        if split_level is None or split_level == "paragraph":
            input_text = [input_text]
        elif split_level == "sentence":
            input_text = spacy.sentence(input_text)
        else:
            raise ValueError(f"split_level must be one of ['paragraph', 'sentence'], got {split_level}")
        for x in input_text:
            x = f"{TASK_PREFIX['qag'] if add_prefix_qg else None}: {x}"
            x = f"{TASK_PREFIX['qag']}: {x}" if add_prefix_qg else x
            output = call_api(
                input_text=x, api_token=api_token, model=model_qg, max_length=max_length, do_sample=do_sample,
                num_beams=num_beams, use_gpu=use_gpu, top_p=top_p)

            for raw_string in output[0]['generated_text'].split(splitting_symbol):
                if len(raw_string.split(answer_prefix)) != 2 or question_prefix not in raw_string:
                    logging.info(f"invalid prediction: {raw_string}")
                else:
                    if raw_string.replace(" ", "").lower() in qa_hash:
                        continue
                    qa_hash.append(raw_string.replace(" ", "").lower())
                    q, a = raw_string.split(answer_prefix)
                    a = re.sub(r'\A\s+', '', a)
                    a = re.sub(r'\s+\Z', '', a)
                    q = q.replace(question_prefix, "")
                    q = re.sub(r'\A\s+', '', q)
                    q = re.sub(r'\s+\Z', '', q)
                    qa.append({'question': q, 'answer': a})
    else:
        if input_answer is None or len(input_answer) == 0:

            assert spacy is not None
            logging.info(f"answer extraction: {spacy.algorithm if model_ae is None else model_ae}")
            if model_ae is None:  # keyword extraction
                input_answer = spacy.keyword(input_text)
            else:
                add_prefix_ae = transformers.AutoConfig.from_pretrained(model_ae).add_prefix if add_prefix_answer is None else add_prefix_answer
                if split_level is None or split_level == "paragraph":
                    batch = [highlight_sentence(input_text, i, TASK_PREFIX['ae'] if add_prefix_ae else None) for i in spacy.sentence(input_text)]
                elif split_level == "sentence":
                    batch = [highlight_sentence(i, i, TASK_PREFIX['ae'] if add_prefix_ae else None) for i in spacy.sentence(input_text)]
                else:
                    raise ValueError(f"split_level must be one of ['paragraph', 'sentence'], got {split_level}")

                output = call_api(
                    input_text=batch,
                    api_token=api_token,
                    model=model_ae,
                    max_length=max_length,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    use_gpu=use_gpu,
                    top_p=top_p)
                input_answer = list(filter(None, [clean(i['generated_text']) for i in output]))  # remove None
                input_answer = list(filter(lambda _x: _x in input_text, input_answer))  # remove answers not in input_text
                if len(input_answer) == 0:
                    raise AnswerNotFoundError(input_text)

        logging.info("generate question")
        input_answer = [input_answer] if type(input_answer) is str else input_answer
        batch = [highlight_sentence(input_text, i, None) for i in input_answer]
        if split_level is not None and split_level == "sentence":
            batch = [[_i for _i in spacy.sentence(i) if _i.count(ADDITIONAL_SP_TOKENS['hl']) == 2] for i in batch]
            batch = [i[0] for i in batch if len(i) > 0]
            if len(batch) == 0:
                raise AnswerNotFoundError(input_text)
        if add_prefix_qg:
            batch = [f"{TASK_PREFIX['qg']}: {i}" for i in batch]
        output = call_api(
            input_text=batch,
            api_token=api_token,
            model=model_qg,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            use_gpu=use_gpu)
        for i, a in zip(output, input_answer):

            q = i['generated_text']
            raw_string = f"{q}, {a}".replace(" ", "").lower()
            if raw_string in qa_hash:
                continue
            qa_hash.append(raw_string)
            qa.append({'question': q, 'answer': a})

        logging.info(f"complete process: {len(output)} qa pairs generated")
    return qa


