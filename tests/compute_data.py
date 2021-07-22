from t5qg.data import get_dataset

for i in ['train', 'test', 'dev']:
    get_dataset('squad', split=i)
    get_dataset('squad', split=i, no_prefix=True)
