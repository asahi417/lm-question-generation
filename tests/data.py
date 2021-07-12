import logging
from tqdm import tqdm
from transformers import AutoTokenizer

from mt5gen.data import get_dataset

level = logging.DEBUG
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

# get_dataset('tydiqa', language=['en'], split='dev', cache_dir='./cache')
tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
for data_name in ['squad']:
    print(data_name)
    data = get_dataset('squad')
    total_in, total_out = [], []
    for i in tqdm(data):
        n_in = len(tokenizer.encode(i['source_text']))
        n_out = len(tokenizer.encode(i['target_text']))
        total_in.append(n_in)
        total_out.append(n_out)

    print('\t input token : {} (<{})'.format(sum(total_in)/len(total_in), max(total_in)))
    print('\t output token: {} (<{})'.format(sum(total_out)/len(total_out), max(total_out)))
