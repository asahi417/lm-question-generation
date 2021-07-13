import json
import logging
import os
import requests
import tarfile
import zipfile
import gzip
from random import shuffle, seed
from typing import List, Dict
from tqdm import tqdm

import gdown
from datasets import load_dataset

from .lm_t5 import TASK_PREFIX, ADDITIONAL_SP_TOKENS
from .sentence_split import SentSplit

__all__ = 'get_dataset'
DEFAULT_CACHE_DIR = '{}/.cache/t5qg'.format(os.path.expanduser('~'))


def get_dataset(name, split: str = 'train', task_type: List or str = 'qg', language: List or str = 'en', cache_dir: str = None):
    language = [language] if type(language) is str else language
    task_type = [task_type] if type(task_type) is str else task_type
    if name == 'squad':
        assert language == ['en'], language
        data = SQuAD(cache_dir).get_data(split, task_type=task_type)
    elif name == 'tydiqa':
        data = TydiQA(cache_dir).get_data(split, language=language, task_type=task_type)
    else:
        raise ValueError('unknown data: {}'.format(name))
    input_texts = [i["source_text"] for i in data]
    output_texts = [i["target_text"] for i in data]
    return input_texts, output_texts


def wget(url, cache_dir: str, gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
        filename = gdrive_filename
    else:
        filename = os.path.basename(url)
        with open('{}/{}'.format(cache_dir, filename), "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    path = '{}/{}'.format(cache_dir, filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)
    return cache_dir


class SQuAD:
    """ Since the original SQuAD's test set is not released, we follow the split used in
    https://arxiv.org/pdf/1705.00106.pdf as same as recent SotA models such as UniLM, ERNIE-GEN, ProhetNet.
    - ref https://github.com/microsoft/unilm/tree/master/unilm-v1#question-generation---squad
    """

    def __init__(self, cache_dir: str):
        self.cache = '{}/data_squad'.format(DEFAULT_CACHE_DIR) if cache_dir is None else cache_dir
        self.output_dir = '{}/processed'.format(self.cache)
        self.sent_splitter = SentSplit()
        self.sp_token_sep = ADDITIONAL_SP_TOKENS['sep']
        self.sp_token_hl = ADDITIONAL_SP_TOKENS['hl']
        logging.info('instantiate SQuAD data processor')

    def get_data(self, split: str = 'train', task_type: List = None):
        assert split in ['train', 'dev', 'test'], split
        output = '{}/{}.jsonl'.format(self.cache, split)
        if os.path.exists(output):
            with open(output, 'r') as f:
                examples = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            path = '{}/{}.json'.format(self.cache, split)
            if not os.path.exists(path):
                wget('https://github.com/xinyadu/nqg/raw/master/data/raw/{}.json'.format(split), cache_dir=self.cache)
            examples = self.process_file(path)
            with open(output, 'w') as f:
                f.write('\n'.join([json.dumps(x) for x in examples]))
        if task_type is None:
            return examples
        else:
            assert all(i in TASK_PREFIX for i in task_type), task_type
            return [i for i in examples if i['task'] in task_type]

    @staticmethod
    def _get_correct_alignment(context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()

    def process_ans_ext(self, context, qas):
        # split into sentences
        sents = self.sent_splitter(context)
        # get positions of the sentences
        positions = []
        prev_end = None
        for i, sent in enumerate(sents):
            if i == 0:
                start, end = 0, len(sent)
            else:
                start, end = (prev_end + 1), (prev_end + len(sent) + 1)
            prev_end = end
            positions.append({'start': start, 'end': end})

        # get answers
        answers = [qa['answers'][0] for qa in qas]
        # get list of answers for each sentence
        sent_answers = []
        for pos, sent in zip(positions, sents):
            target_answers = []
            for ans in answers:
                if ans['answer_start'] in range(pos['start'], pos['end']):
                    target_answers.append(ans['text'].strip())
            sent_answers.append(target_answers)

        # build inputs and targets
        examples = []
        for i, ans in enumerate(sent_answers):
            context = "{}:".format(TASK_PREFIX['ans_ext'])
            if len(ans) == 0:
                continue
            ans = list(set(ans))
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "{0} {1} {0}".format(self.sp_token_hl, sent)
                context = "{} {}".format(context, sent)
                context = context.strip()
            input_text = context
            sep = ' {} '.format(self.sp_token_sep)
            target_text = sep.join(ans) + " " + self.sp_token_sep

            examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})

        return examples

    def process_file(self, filepath):
        """ This function returns the examples in the raw (text) form. """
        logging.info("generating examples from {}".format(filepath))
        # About the task type https://github.com/patil-suraj/question_generation#initial-experiments
        with open(filepath) as f:
            squad = json.load(f)
        examples = []
        for article in tqdm(squad):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                if 'ans_ext' in TASK_PREFIX:
                    examples += self.process_ans_ext(context, paragraph['qas'])
                if 'e2e_qg' in TASK_PREFIX:
                    questions = [qas['question'].strip() for qas in paragraph['qas']]
                    target_text = ' {} '.format(self.sp_token_sep).join(questions) + ' ' + self.sp_token_sep
                    examples.append({"source_text": "{}: {}".format(TASK_PREFIX['e2e_qg'], context),
                                     "target_text": target_text,
                                     "task": "e2e_qg"})
                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    if 'qa' in TASK_PREFIX:
                        examples.append({"source_text": "{}: {}  context: {}".format(TASK_PREFIX['qa'], question, context),
                                         "target_text": qa["answers"][0]["text"].strip(),
                                         "task": "qa"})
                    if 'qg' in TASK_PREFIX:
                        answer_text = qa["answers"][0]['text'].strip()
                        start_pos, end_pos = self._get_correct_alignment(context, qa["answers"][0])
                        que_gen_input = "{0} {1} {2} {1} {3}".format(
                            context[:start_pos], self.sp_token_hl, answer_text, context[end_pos:])
                        examples.append({"source_text": "{}: {}".format(TASK_PREFIX['qg'], que_gen_input),
                                         "target_text": question,
                                         "task": "qg"})
        return examples


class TydiQA:
    """ TODO: To fix the issue of byte position mismatch """

    all_language = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian',
                    'swahili', 'telugu', 'thai']
    all_language_alias = [i[:2] for i in all_language]

    def __init__(self, cache_dir: str):
        self.cache = '{}/data_tydiqa'.format(DEFAULT_CACHE_DIR) if cache_dir is None else cache_dir
        self.output_dir = '{}/processed'.format(self.cache)
        self.sent_splitter = SentSplit()
        self.sp_token_sep = '<sep>'
        self.sp_token_hl = '<hl>'
        logging.info('instantiate TyDIQA data processor')

    def get_data(self, split: str = 'train', language: List = None, task_type: List = None):
        language = self.all_language_alias if language is None else language
        assert split in ['train', 'dev', 'test'], split
        output = '{}/{}.jsonl'.format(self.output_dir, split)
        if os.path.exists(output):
            with open(output, 'r') as f:
                examples = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info("generating examples: {}".format(split))
            dataset = load_dataset("tydiqa", 'primary_task')
            if split == 'dev':
                # exclude YES/NO questions or unanswerable questions
                data = [i for i in dataset['validation'] if -1 not in i['annotations']['minimal_answers_start_byte']]
            else:
                full_ind = []
                for la in language:
                    # exclude YES/NO questions or unanswerable questions
                    ind = [n for n, i in enumerate(dataset['train'])
                           if i['language'][:2] == la and -1 not in i['annotations']['minimal_answers_start_byte']]
                    seed(1)
                    test_size = int(len(ind) * 0.1)
                    shuffle(ind)
                    full_ind += ind[test_size:] if split == 'train' else ind[:test_size]
                full_ind = sorted(full_ind)
                data = [dataset['train'][i] for i in full_ind]
            examples = []
            for _data in tqdm(data):
                examples += self.process_single_data(_data)
            with open(output, 'w') as f:
                f.write('\n'.join([json.dumps(x) for x in examples]))
        return examples

    def process_ans_ext(self, context: str, answer: str):
        sents = self.sent_splitter(context)
        ind = [n for n, s in enumerate(sents) if answer in s][-1]
        end = sum(len(i) for i in sents[:ind + 1]) + 1
        start = 0 if ind == 0 else sum(len(i) for i in sents[:ind]) + 1
        before = ' '.join(sents[:start])
        after = ' '.join(sents[end:])
        sent = sents[start:end]
        return "{0} {1} {2} {1} {3}".format(before, self.sp_token_hl, sent, after)

    def process_single_data(self, paragraph: Dict):
        """ This function returns the examples in the raw (text) form. """
        # About the task type https://github.com/patil-suraj/question_generation#initial-experiments
        tasks = ['qa', 'qg', 'ans_ext', 'e2e_qg']
        context = paragraph["document_plaintext"].strip()
        question = paragraph["question_text"]
        # the answers usually have three candidate, but we simply take the first one as the reference
        answer_start = paragraph['annotations']['minimal_answers_start_byte'][0]
        answer_end = paragraph['annotations']['minimal_answers_end_byte'][0]
        print(answer_start, answer_end)
        answer_text = context.encode("utf-8")[answer_start:answer_end].decode("utf-8", errors="replace")
        answer_text = answer_text.replace('�', '')
        print(answer_text)
        examples = []
        if 'ans_ext' in tasks:
            examples.append({'source_text': "{}: {}".format(TASK_PREFIX['ans_ext'], self.process_ans_ext(context, answer_text)),
                             "target_text": answer_text,
                             "task": "ans_ext"})
        if 'e2e_qg' in tasks:
            examples.append({"source_text": "{}: {}".format(TASK_PREFIX['e2e_qg'], context),
                             "target_text": question + ' ' + self.sp_token_sep,
                             "task": "e2e_qg"})
        if 'qa' in tasks:
            examples.append({"source_text": "{}: {}  context: {}".format(TASK_PREFIX['qa'], question, context),
                             "target_text": answer_text,
                             "task": "qa"})
        if 'qg' in tasks:
            # print(answer_text, answer_start, answer_end)
            before = context.encode("utf-8")[:answer_start].decode("utf-8", errors="replace")
            after = context.encode("utf-8")[answer_end:].decode("utf-8", errors="replace")
            input_text = "{0} {1} {2} {1} {3}".format(before, self.sp_token_hl, answer_text, after)
            examples.append({"source_text": "{}: {}".format(TASK_PREFIX['qg'], input_text),
                             "target_text": question,
                             "task": "qg"})
        # print(examples)
        # input()
        return examples
