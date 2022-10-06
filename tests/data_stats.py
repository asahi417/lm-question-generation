import logging
from tqdm import tqdm
from transformers import AutoTokenizer

from t5qg.data import get_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

# get_dataset('tydiqa', language=['en'], split='dev', cache_dir='./cache')
tokenizer = AutoTokenizer.from_pretrained('t5-small')
for data_name in ['squad']:
    # get_dataset(data_name, split='dev')
    # get_dataset(data_name, split='train')
    in_text, out_text = get_dataset(data_name, split='test', task_type='qg')
    total_in, total_out = [], []
    for _in, _out in tqdm(zip(in_text, out_text)):
        n_in = len(tokenizer.encode(_in))
        n_out = len(tokenizer.encode(_out))
        total_in.append(n_in)
        total_out.append(n_out)

    print(len(in_text))
    for i in sorted(list(set(total_out)), reverse=True):
        print()
        print(i)
        print([b for a, b in zip(total_out, out_text) if a == i])
        if i <= 32:
            break
    print()
    print('\t input token : {} (<{})'.format(sum(total_in)/len(total_in), max(total_in)))
    print('\t output token: {} (<{})'.format(sum(total_out)/len(total_out), max(total_out)))
