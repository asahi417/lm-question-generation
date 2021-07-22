import logging
from tqdm import tqdm
from transformers import AutoTokenizer

from t5qg.data import get_dataset
from t5qg.sentence_split import SentSplit

i, o = get_dataset('squad', split='test', task_type=['qa'])
print(i[:2])
i, o = get_dataset('squad', split='test', task_type=['qg'])
print(i[:2])
i, o = get_dataset('squad', split='test', task_type=['ans_ext'])
print(i[:2])
input()

level = logging.DEBUG
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

# get_dataset('tydiqa', language=['en'], split='dev', cache_dir='./cache')
tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
for data_name in ['squad']:
    print(data_name)
    print(get_dataset(data_name, split='test', no_prefix=False)[0][0])
    print(get_dataset(data_name, split='test', no_prefix=True)[0][0])
    input()
    get_dataset(data_name, split='dev')
    get_dataset(data_name, split='train')
    in_text, out_text = get_dataset(data_name, split='dev', task_type='ans_ext', no_prefix=False, cache_dir='cache')
    total_in, total_out = [], []
    for i in tqdm(in_text):
        input(i)

        # n_in = len(tokenizer.encode(i['source_text']))
        # n_out = len(tokenizer.encode(i['target_text']))
        # total_in.append(n_in)
        # total_out.append(n_out)

    print('\t input token : {} (<{})'.format(sum(total_in)/len(total_in), max(total_in)))
    print('\t output token: {} (<{})'.format(sum(total_out)/len(total_out), max(total_out)))
