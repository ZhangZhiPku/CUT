"""
    Implementation of Matrix Factorization in Recommendation system.

    All algorithm are implemented with tensorflow on gpu and tested.

    Author Phoenix.z 2007 - 2018, all rights reserved.
"""

from Rec.Configurations import *
from Rec.Common.Utilites import *
from Rec.Models.MatrixFactorizationModels import SVD_Rec_Model

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

if __name__ == '__main__':
    raw = mount_dataset()
    processed = data_pre_process(raw)

    data_spliter = KFold(n_splits=5)
    for kidx, (train_idx, test_idx) in enumerate(data_spliter.split(processed)):
        train_dataframe = processed.iloc[train_idx]
        test_dataframe = processed.iloc[test_idx]

        # filter out invalid data
        valid_user_ids = set(train_dataframe['uid'])
        valid_item_ids = set(train_dataframe['iid'])
        test_dataframe['user_valid_tag'] = test_dataframe['uid'].apply(lambda x: x in valid_user_ids)
        test_dataframe['item_valid_tag'] = test_dataframe['iid'].apply(lambda x: x in valid_item_ids)
        test_dataframe = test_dataframe[test_dataframe['user_valid_tag']]
        test_dataframe = test_dataframe[test_dataframe['item_valid_tag']]
        test_dataframe = test_dataframe.drop(['user_valid_tag', 'item_valid_tag'], axis=1)
        debug('invalid data are filtered out. remain %d valid data' % (len(test_dataframe)))

        debug('Running on kflod: %d' % (kidx + 1))
        rec_model = SVD_Rec_Model()
        rec_model.fit(train_dataframe)
        print('finished model training, RMSE: %.4f' %
              mean_squared_error(y_true=test_dataframe['ratings'], y_pred=rec_model.predict(test_dataframe)))
