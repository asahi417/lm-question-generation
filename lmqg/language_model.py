""" T5 model. """
import os
import logging
import pickle
import re
import urllib
from math import exp
from itertools import chain
from typing import List, Dict
from multiprocessing import Pool
from tqdm import tqdm
import torch
from torch.nn import functional
import transformers
from .exceptions import ExceedMaxLengthError, HighlightNotFoundError, AnswerNotFoundError
from .spacy_module import SpacyPipeline

__all__ = ('TransformersQG', 'ADDITIONAL_SP_TOKENS', 'TASK_PREFIX', 'clean', 'internet_connection')

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
TASK_PREFIX = {
    "ae": "extract answers",
    "qg": "generate question",
    "qag": "generate question and answer",
    "qa": "answer question"
}
CE_IGNORE_INDEX = -100
ADDITIONAL_SP_TOKENS = {'hl': '<hl>'}
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '0'))
PARALLEL_PROCESSING = bool(int(os.getenv('PARALLEL_PROCESSING', '0')))
DEFAULT_MODELS = {
    'en': 'lmqg/t5-small-squad-multitask',
    'ja': 'lmqg/mt5-small-jaquad-multitask',
    'de': 'lmqg/mt5-small-dequad-multitask',
    'es': 'lmqg/mt5-small-esquad-multitask',
    'ko': 'lmqg/mt5-small-koquad-multitask',
    'ru': 'lmqg/mt5-small-ruquad-multitask',
    'it': 'lmqg/mt5-small-itquad-multitask',
    'fr': 'lmqg/mt5-small-frquad-multitask',
}


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def clean(string):
    string = re.sub(r'\A\s*', '', string)
    string = re.sub(r'\s*\Z', '', string)
    if len(string) > 0:
        return string
    return None


def internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


def load_language_model(model_name, cache_dir: str = None, use_auth_token: bool = False):
    """ load language model from huggingface model hub """
    # tokenizer
    local_files_only = not internet_connection()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only, use_auth_token=use_auth_token)
    config = transformers.AutoConfig.from_pretrained(
        model_name, local_files_only=local_files_only, cache_dir=cache_dir, use_auth_token=use_auth_token)

    # model
    if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
        model_class = transformers.T5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'mt5':
        model_class = transformers.MT5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'bart':
        model_class = transformers.BartForConditionalGeneration.from_pretrained
    elif config.model_type == 'mbart':
        model_class = transformers.MBartForConditionalGeneration.from_pretrained
    else:
        raise ValueError(f'unsupported model type: {config.model_type}')
    model = model_class(
        model_name, config=config, cache_dir=cache_dir, local_files_only=local_files_only, use_auth_token=use_auth_token)
    # add new special tokens to the tokenizer and the model if they don't have it
    tokenizer.add_special_tokens({'additional_special_tokens': list(ADDITIONAL_SP_TOKENS.values())})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model, config


def label_smoothed_loss(logits, labels, epsilon):
    """ https://github.com/huggingface/transformers/blob/55bb4c06f7be141c6d895dbe1f11018dc8580b2d/src/transformers/trainer_pt_utils.py#L430 """
    log_probs = - functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(CE_IGNORE_INDEX)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels.clamp_min_(0)

    nll_loss = log_probs.gather(dim=-1, index=labels)
    nll_loss.masked_fill_(padding_mask, 0.0)

    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset wrapper converting into tensor """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing. """

    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 max_length_output: int = 34,
                 drop_overflow_error_text: bool = False,
                 skip_overflow_error: bool = False,
                 drop_highlight_error_text: bool = False,
                 prefix_type: str = None,
                 padding: bool = True):
        """ Wrapper of encode_plus for multiprocessing.

        @param tokenizer: transforms.Tokenizer
        @param max_length: Max text length of input.
        @param max_length_output: Max text length of output.
        @param drop_overflow_error_text: If true, return None when the input exceeds the max length.
        @param skip_overflow_error: If true, raise an error when the input exceeds the max length.
        @param drop_highlight_error_text: If true, raise an error when a highlight span is not found in the paragraph.
        @param prefix_type: Either of `qg` or `answer_extraction`, which is to add at the beginning of the text.
        @param padding: Pad the sequence to the max length.
        """
        self.prefix = TASK_PREFIX[prefix_type] if prefix_type is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_output = max_length_output
        # NOTE: for model training, we should drop the exceeded input but not for the evaluation
        self.drop_overflow_error_text = drop_overflow_error_text
        self.skip_overflow_error = skip_overflow_error
        self.drop_highlight_error_text = drop_highlight_error_text
        # truncation should be true for the batch process, but not necessary to process single input
        self.param_in = {'truncation': True, 'max_length': self.max_length}
        self.param_out = {'truncation': True, 'max_length': self.max_length_output}
        if padding:
            self.param_in['padding'] = 'max_length'
            self.param_out['padding'] = 'max_length'

    def __call__(self, inputs):
        return self.encode_plus(*inputs)

    def encode_plus(self, input_sequence: str, output_sequence: str = None, input_highlight: str = None):
        """ encode_plus

        @param input_sequence: Input sequence.
        @param output_sequence: Output sequence.
        @param input_highlight: Sub-sequence of `input_sequence` to be surrounded by <hl>.
        @return: The output of `encode_plus`.
        """

        # add highlight to the input
        if input_highlight is not None:
            position = input_sequence.find(input_highlight)
            if position == -1:
                if self.drop_highlight_error_text:
                    return None
                raise HighlightNotFoundError(input_highlight, input_sequence)
            input_sequence = '{0}{1} {2} {1}{3}'.format(
                input_sequence[:position], ADDITIONAL_SP_TOKENS['hl'], input_highlight,
                input_sequence[position+len(input_highlight):])

        if self.prefix is not None:
            input_sequence = f'{self.prefix}: {input_sequence}'

        # handling overflow text
        # drop_overflow_error_text ==> remove the overflow sentence from input
        # skip_overflow_error ==> keep the overflow sentence
        # none of them ==> raise error
        if self.drop_overflow_error_text or not self.skip_overflow_error:
            if len(self.tokenizer.encode(input_sequence)) > self.max_length:
                if not self.drop_overflow_error_text:  # raise error for overflow text
                    raise ExceedMaxLengthError(self.max_length)
                return None  # remove overflow text
            if output_sequence is not None:
                if len(self.tokenizer.encode(output_sequence)) > self.max_length_output:
                    if not self.drop_overflow_error_text:  # raise error for overflow text
                        raise ExceedMaxLengthError(self.max_length)
                    return None  # remove overflow text
        encode = self.tokenizer.encode_plus(input_sequence, **self.param_in)
        if output_sequence is not None:
            encode['labels'] = self.tokenizer.encode(output_sequence, **self.param_out)
        return encode


class TransformersQG:
    """ Transformers Language Model for Question Generation. """

    def __init__(self,
                 model: str = None,
                 max_length: int = 512,
                 max_length_output: int = 256,
                 cache_dir: str = None,
                 add_prefix: bool = None,
                 language: str = 'en',
                 label_smoothing: float = None,
                 skip_overflow_error: bool = False,
                 drop_overflow_error_text: bool = False,
                 drop_highlight_error_text: bool = False,
                 drop_answer_error_text: bool = False,
                 keyword_extraction_model: str = 'positionrank',
                 use_auth_token: bool = False,
                 end2end_qag_model: bool = None,
                 qa_model: bool = None,
                 answer_extraction_model: bool = None,
                 multitask_qag_model: bool = None):
        """ Transformers Language Model for Question Generation.

        @param model: Model alias or path to local model file.
        @param max_length: Max text length of input.
        @param max_length_output: Max text length of output.
        @param cache_dir: Directory to cache transformers model files.
        @param add_prefix: Whether model uses task-specific prefix (eg. True for T5 but False for BART models).
        @param language: Language alias for SpaCy language-specific pipelines (sentencizer/keyword extraction).
        @param label_smoothing: [Fine-tuning parameter] Label smoothing.
        @param drop_overflow_error_text: If true, return None when the input exceeds the max length.
        @param skip_overflow_error: If true, raise an error when the input exceeds the max length.
        @param drop_highlight_error_text: If true, raise an error when a highlight span is not found in the paragraph.
        @param use_auth_token: [optional] Huggingface transformers argument of `use_auth_token`
        """
        if model is None:
            assert language in DEFAULT_MODELS.keys(), f"Model with language '{language}' is not available. " \
                                                      f"Please choose language from '{DEFAULT_MODELS.keys()}'," \
                                                      f"or specify 'model'."
            model = DEFAULT_MODELS[language]

        # flag if the model is end2end QA generation
        if end2end_qag_model is None:
            self.end2end_qag_model = 'qag' in model.split('-')
        else:
            self.end2end_qag_model = end2end_qag_model

        # flag if the model is question answering model
        if qa_model is None:
            self.qa_model = 'question-answering' in model
        else:
            self.qa_model = qa_model

        # flag if the model is multitask QA generation
        if multitask_qag_model is None:
            self.multitask_qag_model = 'multitask' in model.split('-')
        else:
            self.multitask_qag_model = multitask_qag_model

        # flag if the model is answer extraction model
        if answer_extraction_model is None:
            self.answer_extraction_model = 'answer-extraction' in model
        else:
            self.answer_extraction_model = answer_extraction_model

        self.model_name = model
        self.max_length = max_length
        self.max_length_output = max_length_output
        self.label_smoothing = label_smoothing
        self.drop_overflow_error_text = drop_overflow_error_text
        self.skip_overflow_error = skip_overflow_error
        self.drop_highlight_error_text = drop_highlight_error_text
        self.drop_answer_error_text = drop_answer_error_text
        self.tokenizer, self.model, config = load_language_model(
            self.model_name, cache_dir=cache_dir, use_auth_token=use_auth_token)
        self.add_prefix = config.add_prefix if 'add_prefix' in config.to_dict().keys() else add_prefix
        assert self.add_prefix is not None, '`add_prefix` is required for non-fine-tuned models'
        self.spacy_module = SpacyPipeline(language, keyword_extraction_model)
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.info(f'Model `{self.model_name}`')
        logging.info(f'\t * Num of GPU in use: {torch.cuda.device_count()}')
        logging.info(f'\t * Prefix: {self.add_prefix}')
        logging.info(f'\t * Language: {language} (ignore at the training phase)')

    def generate_qa_end2end(self,
                            list_context: str or List,
                            batch_size: int = None,
                            num_beams: int = 4,
                            cache_path: str = None):
        """ Generate question from paragraph and answer. Note that `list_answer` is needed unless they are already
        highlighted in the `list_context`. eg) "I live in <hl> Tokyo <hl>."

        @param list_context: List of input texts.
        @param batch_size: Batch size.
        @param num_beams: Number of beam for model generation.
        @param cache_path: Path to pre-compute features.
        @return: List of generated sentences.
        """
        assert self.end2end_qag_model, "`generate_qa_end2end` is available for end2end_qag_model"
        splitting_symbol = '|'
        question_prefix = "question: "
        answer_prefix = ", answer: "
        prefix_type = 'qag' if self.add_prefix else None
        single_input = False
        if type(list_context) is str:
            list_context = [list_context]
            single_input = True
        output = self.generate_prediction(
            list_context, prefix_type=prefix_type, cache_path=cache_path, num_beams=num_beams, batch_size=batch_size
        )

        def format_qa(list_raw_string):
            tmp = []
            for raw_string in list_raw_string:
                if question_prefix not in raw_string or answer_prefix not in raw_string:
                    logging.info(f"invalid prediction: {raw_string}")
                else:
                    q, a = raw_string.split(answer_prefix)
                    a = re.sub(r'\A\s+', '', a)
                    a = re.sub(r'\s+\Z', '', a)
                    q = q.replace(question_prefix, "")
                    q = re.sub(r'\A\s+', '', q)
                    q = re.sub(r'\s+\Z', '', q)
                    tmp.append((q, a))
            return tmp

        output = [format_qa(o.split(splitting_symbol)) for o in output]
        if single_input:
            return output[0]
        return output

    def generate_qa(self,
                    list_context: str or List,
                    batch_size: int = None,
                    num_beams: int = 4,
                    cache_path: str = None,
                    answer_model: str = None,
                    num_questions: int = None,
                    sentence_level: bool = False):
        """ Generate question given context.

        @param list_context: Input text.
        @param batch_size: Batch size.
        @param num_beams: Number of beam for model generation.
        @param cache_path: Path to pre-compute features.
        @param answer_model: Type of answer prediction model (`keyword_extraction`. `language_model`).
            - `keyword`: Keyword extraction model extracts top-n keyword.
            - `language_model`: LM predicts answers (Model should have been finetuned on `answer_extraction`.
        @param num_questions: Max number of questions.
        @param sentence_level: Run prediction on each sentence of the context independently to reduce complexity.
        @return: List of generated sentences.
        """
        if self.end2end_qag_model:
            return self.generate_qa_end2end(list_context, batch_size, num_beams, cache_path)
        output = []
        single_input = False
        if type(list_context) is str:
            list_context = [list_context]
            single_input = True
        for c in list_context:
            list_answer = self.generate_a(
                c, batch_size=batch_size, num_beams=num_beams, cache_path=cache_path,
                sentence_level=sentence_level, answer_model=answer_model, num_questions=num_questions
            )
            if list_answer is None:
                output.append(None)
                continue
            list_context = [c] * len(list_answer)
            logging.info('running model for `qg`')
            list_question = self.generate_q(
                list_context,
                list_answer=list_answer,
                batch_size=batch_size,
                cache_path=cache_path,
                num_beams=num_beams,
                sentence_level=sentence_level
            )
            assert len(list_answer) == len(list_question)
            output.append(list(zip(list_question, list_answer)))
        if single_input:
            return output[0]
        return output

    def generate_a(self,
                   context: str,
                   batch_size: int = None,
                   num_beams: int = 4,
                   cache_path: str = None,
                   sentence_level: bool = False,
                   answer_model: str = None,
                   num_questions: int = None):
        """ Generate answers from each sentence.

        @param context: Input text.
        @param batch_size: Batch size.
        @param num_beams: Number of beam for model generation.
        @param cache_path: Path to pre-compute features.
        @param sentence_level: Run prediction on each sentence of the context independently to reduce complexity.
        @param num_questions: Max number of questions.
        @param answer_model: Type of answer prediction model (`keyword_extraction`. `language_model`).
            - `keyword`: Keyword extraction model extracts top-n keyword.
            - `ner`: Named-entity recognition.
            - `language_model`: LM predicts answers (Model should have been finetuned on `answer_extraction`.
        @return: List of generated answers.
        """
        assert not self.end2end_qag_model, "end2end qag model can not generate answer only"
        if answer_model is None:
            answer_model = 'language_model' if self.multitask_qag_model else 'keyword_extraction'

        logging.info(f'running model for `answer_extraction`: {answer_model}')
        if answer_model == 'keyword_extraction':
            num_questions = 10 if num_questions is None else num_questions
            return self.spacy_module.keyword(context, num_questions)
        elif answer_model == 'ner':
            return self.spacy_module.ner(context, num_questions)
        elif answer_model == 'language_model':
            assert self.multitask_qag_model or self.answer_extraction_model,\
                f"The model {self.model_name} is not fine-tuned for answer extraction, " \
                f"and not able to get answer. Try `answer_model = 'keyword_extraction'` instead."

            list_sentence = self.spacy_module.sentence(context)  # split into sentence
            list_input = [context] * len(list_sentence)
            if sentence_level:
                list_input = list_sentence
            answer = self.generate_prediction(
                list_input,
                highlights=list_sentence,
                prefix_type='ae' if self.add_prefix else None,
                cache_path=cache_path,
                num_beams=num_beams,
                batch_size=batch_size)
            answer = [clean(i) for i in answer]
            answer = list(filter(None, answer))  # remove None
            answer = list(filter(lambda x: x in context, answer))  # remove answers not in context (should not be happened though)
            if len(answer) == 0:
                if self.drop_answer_error_text:
                    return None
                raise AnswerNotFoundError(context)
            if num_questions is not None:
                answer = answer[:min(num_questions, len(answer))]
            return answer
        else:
            raise ValueError(f'unknown answer model: {answer_model}')

    def generate_q(self,
                   list_context: str or List,
                   list_answer: List = None,
                   batch_size: int = None,
                   num_beams: int = 4,
                   cache_path: str = None,
                   sentence_level: bool = False):
        """ Generate question from paragraph and answer. Note that `list_answer` is needed unless they are already
        highlighted in the `list_context`. eg) "I live in <hl> Tokyo <hl>."

        @param list_context: List of input texts.
        @param list_answer: List of answers in the `list_context` that are highlighted by <hl>.
        @param batch_size: Batch size.
        @param num_beams: Number of beam for model generation.
        @param cache_path: Path to pre-compute features.
        @param sentence_level: Run prediction on each sentence of the context independently to reduce complexity.
        @return: List of generated sentences.
        """
        assert not self.end2end_qag_model, "end2end qag model can not generate question only"
        assert not self.answer_extraction_model, "model is not fine-tuned for QG"
        assert not self.qa_model, "model is not fine-tuned for QG"
        single_input = False
        if type(list_context) is str:
            list_context = [list_context]
            single_input = True
        output = self.generate_prediction(
            list_context,
            highlights=list_answer,
            prefix_type='qg' if self.add_prefix else None,
            cache_path=cache_path,
            num_beams=num_beams,
            batch_size=batch_size,
            sentence_level=sentence_level
        )
        if single_input:
            return output[0]
        return output

    def answer_q(self,
                 list_context: str or List,
                 list_question: str or List,
                 batch_size: int = None,
                 num_beams: int = 4,
                 cache_path: str = None):
        assert self.qa_model, "model is not fine-tuned for QA"
        assert type(list_context) is type(list_question), "invalid input"
        single_input = False
        if type(list_context) is str:
            list_context = [list_context]
            list_question = [list_question]
            single_input = True
        output = self.generate_prediction(
            [f"question: {q}, context: {c}" for q, c in zip(list_question, list_context)],
            prefix_type='qa' if self.add_prefix else None,
            cache_path=cache_path,
            num_beams=num_beams,
            batch_size=batch_size,
        )
        if single_input:
            return output[0]
        return output

    def generate_prediction(self,
                            inputs: List,
                            highlights: List or None = None,
                            prefix_type: str = None,
                            num_beams: int = 4,
                            batch_size: int = None,
                            cache_path: str = None,
                            sentence_level: bool = False):
        """ General method to generate model prediction

        @param inputs: List of input sequences.
        @param highlights: List of sub-sequences from list_context to be highlighted by <hl>.
        @param batch_size: Batch size.
        @param num_beams: Number of beam for model generation.
        @param cache_path: Path to pre-compute features.
        @param prefix_type: Either of `qg` or `answer_extraction`, which is to add at the beginning of the text.
        @return: List of generated sequences.
        """
        self.eval()

        if sentence_level:
            assert highlights is not None, '`sentence_level` needs `highlights`.'
            assert len(highlights) == len(inputs), str([len(highlights), len(inputs)])
            list_sentence = []
            for context, answer in zip(inputs, highlights):
                s = [sentence for sentence in self.spacy_module.sentence(context) if answer in sentence]
                list_sentence.append(s[0] if len(s) != 0 else context)
            inputs = list_sentence

        assert type(inputs) is list, inputs
        encode_list = self.text_to_encode(inputs, highlights=highlights, prefix_type=prefix_type, cache_path=cache_path)
        loader = self.get_data_loader(encode_list, batch_size=batch_size)
        outputs = []
        for encode in loader:
            with torch.no_grad():
                if 'labels' in encode:
                    encode.pop('labels')
                encode = {k: v.to(self.device) for k, v in encode.items()}
                encode['max_length'] = self.max_length_output
                encode['num_beams'] = num_beams
                tensor = self.model.module.generate(**encode) if self.parallel else self.model.generate(**encode)
                outputs += self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return outputs

    def encode_to_loss(self, encode: Dict):
        """ Transform encoded features to loss value for model finetuning.

        @param encode: Encoded feature.
        @return: Loss value.
        """
        assert 'labels' in encode
        output = self.model(**{k: v.to(self.device) for k, v in encode.items()})
        if self.label_smoothing is None or self.label_smoothing == 0.0:
            return output['loss'].mean() if self.parallel else output['loss']
        else:
            return label_smoothed_loss(output['logits'], encode['labels'].to(self.device), self.label_smoothing)

    def text_to_encode(self,
                       inputs,
                       outputs: List = None,
                       highlights: List = None,
                       prefix_type: str = None,
                       cache_path: str = None):
        """ Transform texts into encoded features.

        @param inputs: List of input sequences.
        @param outputs: List of output sequences.
        @param highlights: List of sub-sequences from `inputs` to be highlighted by <hl>.
        @param prefix_type: Either of `qg` or `answer_extraction`, which is to add at the beginning of the text.
        @param cache_path: Path to pre-compute features.
        @return: List of encoded feature.
        """
        if cache_path is not None and os.path.exists(cache_path):
            logging.info(f'loading preprocessed feature from {cache_path}')
            return pickle_load(cache_path)
        outputs = [None] * len(inputs) if outputs is None else outputs
        highlights = [None] * len(inputs) if highlights is None else highlights
        assert len(outputs) == len(inputs) == len(highlights), str([len(outputs), len(inputs), len(highlights)])
        data = list(zip(inputs, outputs, highlights))

        # process in parallel/single
        config = {'tokenizer': self.tokenizer, 'max_length': self.max_length, 'prefix_type': prefix_type,
                  'max_length_output': self.max_length_output, 'drop_overflow_error_text': self.drop_overflow_error_text,
                  'skip_overflow_error': self.skip_overflow_error, 'drop_highlight_error_text': self.drop_highlight_error_text,
                  'padding': False if len(data) == 1 else True}

        logging.info(f'encode all the data       : {len(data)}')
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if PARALLEL_PROCESSING:
            pool = Pool()
            out = pool.map(EncodePlus(**config), data)
            pool.close()
            out = list(filter(None, out))  # remove overflow text
        else:
            f = EncodePlus(**config)
            out = []
            files = []
            for i in tqdm(data):
                e = f(i)
                if e is not None:  # remove overflow text
                    out.append(e)
                if len(out) > 40000 and cache_path is not None:
                    pickle_save(out, f'{cache_path}.tmp{len(files)}')
                    files.append(f'{cache_path}.tmp{len(files)}')
                    out = []
            if len(out) > 0 and cache_path is not None:
                pickle_save(out, f'{cache_path}.tmp{len(files)}')
                files.append(f'{cache_path}.tmp{len(files)}')
            if len(files) > 0:
                out = list(chain(*[pickle_load(i) for i in files]))
        logging.info(f'after remove the overflow : {len(out)}')
        # cache the encoded data
        if cache_path is not None:
            pickle_save(out, cache_path)
            logging.info(f'preprocessed feature is saved at {cache_path}')
        return out

    def save(self, save_dir):
        """ Save model.

        @param save_dir: Directory to save model related file.
        """

        def model_state(model):
            if self.parallel:
                return model.module
            return model

        logging.info('saving model')
        model_state(self.model).config.update({'add_prefix': self.add_prefix})
        model_state(self.model).save_pretrained(save_dir)
        logging.info('saving tokenizer')
        self.tokenizer.save_pretrained(save_dir)

    @staticmethod
    def get_data_loader(encode_list, batch_size: int = None, shuffle: bool = False, drop_last: bool = False):
        """ Get torch.utils.data.DataLoader instance.

        @param encode_list: List of encoded features.
        @param batch_size: Batch size.
        @param shuffle: Shuffle data.
        @param drop_last: Drop residual batch.
        @return: torch.utils.data.DataLoader
        """
        batch_size = len(encode_list) if batch_size is None else batch_size
        params = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=NUM_WORKERS)
        return torch.utils.data.DataLoader(Dataset(encode_list), **params)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def get_perplexity(self,
                       list_context: List,
                       list_answer: List = None,
                       list_question: List = None,
                       list_questions_answers: List = None,
                       batch_size: int = None,
                       target_output: str = 'answer'):
        if target_output == 'questions_answers':
            assert self.end2end_qag_model, "this method is for end2end QAG model"
            assert list_questions_answers is not None
            if self.add_prefix:
                list_input = [f"{TASK_PREFIX['qag']}: {i}" for i in list_context]
            else:
                list_input = list_context
            list_output = list_questions_answers
        elif target_output == 'question':
            assert not self.end2end_qag_model, "this method is not for end2end QAG model"
            assert list_answer is not None, '`list_answer` is required to compute ppl on question'
            assert list_question is not None, '`list_question` is required to compute ppl on question'
            prefix_type = 'qg' if self.add_prefix else None
            list_input = []
            list_output = list_question
            for context, answer in zip(list_context, list_answer):
                position = context.find(answer)
                if position == -1:
                    raise HighlightNotFoundError(answer, context)
                tmp = '{0}{1} {2} {1}{3}'.format(
                    context[:position], ADDITIONAL_SP_TOKENS['hl'], answer, context[position + len(answer):])
                if prefix_type is None:
                    list_input.append(tmp)
                else:
                    list_input.append(f'{TASK_PREFIX[prefix_type]}: {tmp}')
        elif target_output == 'answer':
            assert not self.end2end_qag_model, "this method is not for end2end QAG model"
            assert self.multitask_qag_model, "perplexity on answer is available only for multitask models"
            assert list_answer is not None
            if not self.add_prefix:
                raise ValueError(f"The model {self.model_name} is not fine-tuned for answer extraction, "
                                 f"and not able to get answer. Try `answer_model = 'keyword_extraction'` instead.")

            # get perplexity for the answer given the paragraph
            prefix_type = 'ae'  # multitask model should have a prefix
            list_input = []
            list_output = list_answer
            for context, answer in zip(list_context, list_answer):
                # split into sentences and find a sentence that contain the answer
                sentences = [s for s in self.spacy_module.sentence(context) if answer in s]
                if len(sentences) == 0:
                    sentence = context
                else:
                    sentence = sentences[0]
                position = context.find(sentence)
                if position == -1:
                    raise HighlightNotFoundError(sentence, context)
                tmp = '{0}{1} {2} {1}{3}'.format(
                    context[:position], ADDITIONAL_SP_TOKENS['hl'], sentence, context[position + len(sentence):])
                list_input.append(f'{TASK_PREFIX[prefix_type]}: {tmp}')
        else:
            raise ValueError(f'invalid target_output {target_output}')
        return self.compute_decoder_perplexity(
            src_texts=list_input, tgt_texts=list_output, batch=batch_size
        )

    def compute_decoder_perplexity(self,
                                   src_texts: str or List,
                                   tgt_texts: str or List,
                                   batch: int = None,
                                   fill_error=None):
        """ Compute the perplexity on the decoder of the seq2seq model. """
        self.eval()
        single_input = False

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')

        if type(src_texts) == str:
            src_texts = [src_texts]
            single_input = True
        if type(tgt_texts) == str:
            tgt_texts = [tgt_texts]
        if fill_error is not None:
            raise NotImplementedError("TODO find nans and remove them and return to the index")

        assert len(src_texts) == len(tgt_texts), f"{len(src_texts)} == {len(tgt_texts)}"
        batch = len(tgt_texts) if batch is None else batch
        batch_id = list(range(0, len(src_texts), batch)) + [len(tgt_texts)]
        batch_id = list(zip(batch_id[:-1], batch_id[1:]))
        loss_list = []
        with torch.no_grad():
            for s, e in tqdm(batch_id):
                model_inputs = self.tokenizer(src_texts[s:e], return_tensors='pt', padding=True, truncation=True)
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(tgt_texts[s:e], return_tensors='pt', padding=True, truncation=True)

                model_inputs["labels"] = labels["input_ids"]
                out = self.model(**{k: v.to(self.device) for k, v in model_inputs.items()})
                loss = loss_fct(
                    out['logits'].view(-1, out['logits'].size(-1)),
                    model_inputs["labels"].view(-1).to(self.device)
                )
                loss_aligned = loss.view(out['logits'].size(0), out['logits'].size(1))
                loss_final = loss_aligned.mean(-1)
                loss_list += loss_final.cpu().numpy().tolist()
        ppl = [exp(i) for i in loss_list]
        assert len(ppl) == len(src_texts)
        if single_input:
            return ppl[0]
        return ppl




