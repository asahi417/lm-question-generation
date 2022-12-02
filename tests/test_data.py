import os
import json
import shutil
from glob import glob
from lmqg import get_dataset, get_reference_files
from tqdm import tqdm


def add_qa_input(_dict):
    _dict['paragraph_question'] = f"question: {_dict['question']}, context: {_dict['paragraph']}"
    return _dict


data_list = [
    "lmqg/qg_squad",
    "lmqg/qg_jaquad",
    "lmqg/qg_esquad",
    "lmqg/qg_koquad",
    "lmqg/qg_dequad",
    "lmqg/qg_itquad",
    "lmqg/qg_ruquad",
    "lmqg/qg_frquad"
]
data_list_domains = [
    ["lmqg/qg_squadshifts", ["all", "amazon", "reddit", "new_wiki", "nyt"]],
    ["lmqg/qg_subjqa", ["all", "books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"]]
]

for data in data_list:
    get_dataset(data)
    get_reference_files(data)
for data, domains in data_list_domains:
    for d in domains:
        get_dataset(data, d)
        get_reference_files(data, d)
# for data in data_list + [k[0] for k in data_list_domains]:
#
#     os.system(f"git clone https://huggingface.co/datasets/{data}")
#     for _file in tqdm(glob(f"{os.path.basename(data)}/data/processed/*.jsonl")):
#         with open(_file) as f:
#             jsonlines = [add_qa_input(json.loads(i)) for i in f.read().split('\n') if len(i) >0]
#         with open(_file, 'w') as f:
#             f.write('\n'.join([json.dumps(i) for i in jsonlines]))
#     with open(f"{os.path.basename(data)}/{os.path.basename(data)}.py") as f:
#         tmp = f.read()
#         if '"paragraph_question": datasets.Value("string"),' not in tmp:
#             tmp = tmp.replace('"answer": datasets.Value("string"),', '"answer": datasets.Value("string"), "paragraph_question": datasets.Value("string"),')
#         tmp = '\n'.join(['_VERSION = "5.0.1"' if i.startswith('_VERSION') else i for i in tmp.split('\n')])
#         tmp = tmp.replace('raw/main/data/processed', 'resolve/main/data/processed')
#
#     with open(f"{os.path.basename(data)}/{os.path.basename(data)}.py", 'w') as f:
#         f.write(tmp)
#     # os.remove(f"{os.path.basename(data)}/generate_reference_files.py")
#     # shutil.rmtree(f"{os.path.basename(data)}/reference_files")
#     os.system(f"cd {os.path.basename(data)} && git lfs install && git lfs track data/processed/*.jsonl && git add . && git commit -m 'update' && git push && cd ../")
#     shutil.rmtree(os.path.basename(data))
#

data_qag = [
    "lmqg/qag_tweetqa",
    "lmqg/qag_squad"
]

data_qa = [
    "lmqg/qa_harvesting_from_wikipedia",
    "lmqg/qa_harvesting_from_wikipedia_pseudo",
    "lmqg/qa_squad"
]
data_qa_domain = [
    "lmqg/qa_squadshifts",
    "lmqg/qa_squadshifts_pseudo"
]
for data in data_qag:

    os.system(f"git clone https://huggingface.co/datasets/{data}")
    for _file in tqdm(glob(f"{os.path.basename(data)}/data/processed/*.jsonl")):
        with open(_file) as f:
            jsonlines = [add_qa_input(json.loads(i)) for i in f.read().split('\n') if len(i) >0]
        with open(_file, 'w') as f:
            f.write('\n'.join([json.dumps(i) for i in jsonlines]))
    with open(f"{os.path.basename(data)}/{os.path.basename(data)}.py") as f:
        tmp = f.read()
        if '"paragraph_question": datasets.Value("string"),' not in tmp:
            tmp = tmp.replace('"answer": datasets.Value("string"),', '"answer": datasets.Value("string"), "paragraph_question": datasets.Value("string"),')
        tmp = '\n'.join(['_VERSION = "5.0.1"' if i.startswith('_VERSION') else i for i in tmp.split('\n')])
        tmp = tmp.replace('raw/main/data/processed', 'resolve/main/data/processed')

    with open(f"{os.path.basename(data)}/{os.path.basename(data)}.py", 'w') as f:
        f.write(tmp)
    # os.remove(f"{os.path.basename(data)}/generate_reference_files.py")
    # shutil.rmtree(f"{os.path.basename(data)}/reference_files")
    os.system(f"cd {os.path.basename(data)} && git lfs install && git lfs track data/processed/*.jsonl && git add . && git commit -m 'update' && git push && cd ../")
    shutil.rmtree(os.path.basename(data))

