"""
    Implementation of Matrix Factorization in Recommendation system.

    All algorithm are implemented with tensorflow on gpu and tested.

    Author Phoenix.z 2007 - 2018, all rights reserved.
"""

from Rec.Configurations import *
from Rec.Common.Utilites import *
from Rec.Models.SequentialModels import UPS_Cliassifler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def generate_data():
    import pickle

    try:
        debug('System trying to load cached data...')
        with open(file=DATA_CACHE_FILE, mode='rb') as file:
            dataframe, n_items, n_users = pickle.load(file)
        debug('Cached data load successfully.')
    except Exception as e:
        debug('Cache file does not exist, System trying to make new copy of dataset...')
        dataframe, n_items, n_users = mount_dataset()
        with open(file=DATA_CACHE_FILE, mode='wb') as file:
            pickle.dump([dataframe, n_items, n_users], file)
    return dataframe, n_items, n_users

if __name__ == '__main__':
    dataframe, n_items, n_users = generate_data()
    # filter out unrelated columns
    dataframe['X'] = dataframe['enTollLaneId']
    dataframe['y'] = dataframe['user_type']  # -1 if label is {1, 2}
    dataframe = dataframe[['X', 'y']]

    data_spliter = KFold(n_splits=3, shuffle=True)
    for kidx, (train_idx, validation_idx) in enumerate(data_spliter.split(dataframe)):
        train_dataframe = dataframe.iloc[train_idx].copy()
        validation_dataframe = dataframe.iloc[validation_idx].copy()

        sequence_length = min(train_dataframe['X'].apply(
            lambda x: len(x)).quantile(0.95), MAXIMUM_SEQUENCE_LENGTH)
        print('Model input sequence length has been automatically set as %d' % sequence_length)

        model = UPS_Cliassifler(n_items=n_items, sequence_length=sequence_length, rank=100)
        model.pretrain(train_dataframe['X'])
        model.finetune(dataframe=train_dataframe)

        print(validation_dataframe['y'])

        prediction = model.predict(dataframe=validation_dataframe)
        print('model training finished at iteration %d, model auc score: %.4f' %
              (kidx + 1, roc_auc_score(y_true=validation_dataframe['y'],
                                       y_score=prediction)))