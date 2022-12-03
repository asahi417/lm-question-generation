from lmqg import get_dataset, get_reference_files


data_list = [
    "lmqg/qg_squad",
    "lmqg/qg_jaquad",
    "lmqg/qg_esquad",
    "lmqg/qg_koquad",
    "lmqg/qg_dequad",
    "lmqg/qg_itquad",
    "lmqg/qg_ruquad",
    "lmqg/qg_frquad",
    "lmqg/qg_tweetqa"
]
data_list_domains = [
    ["lmqg/qg_squadshifts", ["all", "amazon", "reddit", "new_wiki", "nyt"]],
    ["lmqg/qg_subjqa", ["all", "books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"]]
]

data_qag = [
    "lmqg/qag_tweetqa",
    "lmqg/qag_squad"
]

for data in data_list:
    get_dataset(data, input_type='paragraph_question', output_type='answer')
    get_reference_files(data)

for data in data_qag:
    get_dataset(data, input_type='paragraph', output_type='questions_answers')
    get_reference_files(data)

for data, domains in data_list_domains:
    for d in domains:
        get_dataset(data, d, input_type='paragraph', output_type='answer')
        get_reference_files(data, d)


