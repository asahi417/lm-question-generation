from lmqg.data import get_dataset, get_nqg_evaluation_reference_files

print(get_nqg_evaluation_reference_files())
data = 'squad'

for i in ['train', 'test', 'dev']:
    get_dataset(data, split=i)
    get_dataset(data, split=i, no_prefix=True)
