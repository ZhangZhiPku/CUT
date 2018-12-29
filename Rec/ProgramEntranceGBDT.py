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
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm


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
    dataframe['X'] = dataframe['X'].apply(lambda x: ''.join([str(_) + ' ' for _ in x]))
    data_spliter = KFold(n_splits=4, shuffle=True)
    for kidx, (train_idx, validation_idx) in enumerate(data_spliter.split(dataframe)):
        train_dataframe = dataframe.iloc[train_idx].copy()
        validation_dataframe = dataframe.iloc[validation_idx].copy()
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.1,
                                     min_df=1e-5, analyzer='word')
        print('feature extraction fitting on training data.')
        vectorizer.fit(train_dataframe['X'])

        train_X = vectorizer.transform(train_dataframe['X'])
        validation_X = vectorizer.transform(validation_dataframe['X'])

        model = lightgbm.LGBMClassifier(objective='binary', n_estimators=300,
                                        class_weight='balanced', reg_lambda=1e-2,
                                        learning_rate=0.15, num_leaves=31)
        model.fit(train_X, train_dataframe['y'])

        print('model training finished at iteration %d, model auc score: %.4f' %
              (kidx + 1, roc_auc_score(y_true=validation_dataframe['y'],
                                       y_score=model.predict_proba(validation_X)[:, 1])))