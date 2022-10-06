import json
import logging
import os
import re
from random import shuffle, seed
from tqdm import tqdm

from datasets import load_dataset

from t5qg import sentence_split, wget

SPLITTER = sentence_split.SentSplit()


def cleaner(string):
    string = re.sub(r'\[\d*\]', '', string)
    string = re.sub(r'\s+', ' ', string)
    string = re.sub(r'\A\s*', '', string)
    string = re.sub(r'\s*\Z', '', string)
    return string


def check_answer(context, answer):
    if answer not in context:
        logging.warning('answer not found: \n - answer: {}\n - context: {}'.format(answer, context))
        return False
    return True
    # sents = SPLITTER(context)
    # ind_candidate = [n for n, s in enumerate(sents) if answer in s]
    # if len(ind_candidate) == 0:
    #     logging.warning('answer not found: \n - answer: {}\n - context: {}'.format(answer, context))
    #     return False
    # return True


def process_tydiqa(cache_dir: str = 'cache/tydiqa'):
    language = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 'korean', 'russian', 'swahili', 'telugu']
    stats = {}
    dataset = load_dataset('tydiqa', 'secondary_task')
    for la in language:
        data_dev = [i for i in dataset['validation'] if i['id'].split('-')[0] == la]
        data_train_tmp = [i for i in dataset['train'] if i['id'].split('-')[0] == la]
        seed(1)
        shuffle(data_train_tmp)
        data_train = data_train_tmp[len(data_dev):]
        data_test = data_train_tmp[:len(data_dev)]
        data = {'train': data_train, 'dev': data_dev, 'test': data_test}
        stats[la] = {k: 0 for k in data.keys()}
        for k, v in data.items():
            logging.info('processing {}/{}'.format(k, la))
            output = '{}/{}.{}.jsonl'.format(cache_dir, la, k)
            os.makedirs(cache_dir, exist_ok=True)
            with open(output, 'w') as outfile:
                logging.info("\t * generating examples: {}".format(k))
                for i in tqdm(v):
                    if i['id'].split('-')[0] == la:
                        answer = cleaner(i['answers']['text'][0])
                        context = cleaner(i['context'])
                        question = cleaner(i['question'])
                        if check_answer(context, answer):
                            json.dump({'answer': answer, 'context': context, 'question': question}, outfile)
                            outfile.write('\n')
                            stats[la][k] += 1
    logging.info(stats)
    with open('{}/statistics.json'.format(cache_dir), 'w') as f:
        json.dump(stats, f)


def process_squad(cache_dir: str = 'cache/squad'):
    stats = {}
    for split in ['train', 'dev', 'test']:
        stats[split] = 0
        logging.info('processing {}'.format(split))
        path = '{}/{}.json'.format(cache_dir, split)
        if not os.path.exists(path):
            wget('https://github.com/xinyadu/nqg/raw/master/data/raw/{}.json'.format(split), cache_dir=cache_dir)
        with open(path) as f:
            data = json.load(f)
        output = '{}/{}.jsonl'.format(cache_dir, split)
        os.makedirs(cache_dir, exist_ok=True)
        with open(output, 'w') as outfile:
            for article in tqdm(data):
                for paragraph in article["paragraphs"]:
                    context = cleaner(paragraph["context"])
                    for qa in paragraph["qas"]:
                        question = cleaner(qa["question"])
                        answer = cleaner(qa["answers"][0]['text'])
                        if check_answer(context, answer):
                            json.dump({'answer': answer, 'context': context, 'question': question}, outfile)
                            outfile.write('\n')
                            stats[split] += 1
        os.remove('{}/*.json'.format(cache_dir))
    logging.info(stats)
    with open('{}/statistics.json'.format(cache_dir), 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    process_tydiqa()
    process_squad()
