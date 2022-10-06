from t5qg.data import get_dataset

# data = 'squad'
for data in ['squad']:
    for i in ['train', 'test', 'dev']:
        get_dataset(data, split=i)
        get_dataset(data, split=i, no_prefix=True)
