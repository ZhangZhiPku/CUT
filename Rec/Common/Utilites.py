import tqdm
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from Rec.Configurations import DEBUG_MODE


def debug(info):
    if DEBUG_MODE:
        time_for_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time_for_now + '\t' + info)


def mount_dataset(dataset_file_path='data/u.data'):
    """
    load dataset from disk.

    By default we use movie-lens 10k dataset,
    you have to rewrite this function if you want to switch to another dataset.

    :return: list of rating tuples: (uid, iid, ratings, timestamp)
    """

    import os
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(
            'Data is not found, please check target file.\n {0}'.format(dataset_file_path))

    debug('Now system is trying load dataset from file.')

    try:
        with open(dataset_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            rating_tuples = [tuple(line.split('\t')) for line in tqdm.tqdm(file)]
            debug('Data set successfully loaded, with data length {0}'.format(len(rating_tuples)))
            return rating_tuples
    except Exception as e:
        raise e


def data_pre_process(data):
    """
    Do pre-processing with loaded data.

    :param data: loaded raw data.

    :return: processed data.
    """

    debug('Now system is trying to do pre-processing on data set.')

    processed = pd.DataFrame(data, columns=['uid', 'iid', 'ratings', 'timestamp'])

    # convert data type to desired one.
    processed['uid'] = processed['uid'].astype(dtype=np.int32)
    processed['iid'] = processed['iid'].astype(dtype=np.int32)
    processed['ratings'] = processed['ratings'].astype(dtype=np.float32)
    processed['timestamp'] = processed['timestamp'].astype(dtype=np.int32)
    processed = processed.sort_values(by=['timestamp'])

    debug('Now system is dealing with id mapping')
    distinct_user_ids = processed['uid'].unique()

    # transform item ids into frequency desent order
    distinct_item_ids = processed.groupby(['iid']).count()
    distinct_item_ids = distinct_item_ids.sort_values(['uid'], ascending=False)
    distinct_item_ids = distinct_item_ids.index

    # mapping user id and item id into continuous vector space
    user_mapping = {uid: pos for pos, uid in enumerate(distinct_user_ids)}
    item_mapping = {iid: pos for pos, iid in enumerate(distinct_item_ids)}

    n_items, n_users = len(item_mapping), len(user_mapping)
    processed['uid'] = processed['uid'].apply(lambda x: user_mapping[x])
    processed['iid'] = processed['iid'].apply(lambda x: item_mapping[x])

    debug('Now system is trying make sequential data.')
    sequential_dict = defaultdict(list)
    for idx, row in tqdm.tqdm(processed.iterrows(), total=len(processed)):
        sequential_dict[int(row.uid)].append(int(row.iid))

    return sequential_dict.items(), n_items, n_users


class Training_Data_Batcher():
    """
    Training data batcher
    """
    def __init__(self, dataframe, batch_size=128, reshuffle=True):
        if isinstance(dataframe, pd.DataFrame) or isinstance(dataframe, pd.Series):
            if reshuffle:
                self.data = dataframe.sample(n=len(dataframe))
            else:
                self.data = dataframe

            self.batch_size = batch_size
            self.batch_start_idx = 0
            self.n_data = len(self.data)
            self.reshuffle = reshuffle
        else:
            raise Exception(
                'Input variable dataframe must be one of pandas dataframe, but {0} are given'.format(
                    type(dataframe)))

    def make_batch(self):
        ret = self.data.iloc[self.batch_start_idx:
                             self.batch_start_idx + self.batch_size]
        self.batch_start_idx += self.batch_size
        if self.batch_start_idx >= self.n_data:
            if self.reshuffle:
                self.data = self.data.sample(n=self.n_data)
            self.batch_start_idx = 0
        return ret

    def make_walk_through(self, batchsize=128):
        ret, batch_start_idx = [], 0
        while batch_start_idx < self.n_data:
            ret.append(self.data.iloc[batch_start_idx: batch_start_idx + batchsize])
            batch_start_idx += batchsize
        return ret


class Training_Helper():
    """
    Tensorflow Model Training Delegate

    That is something like keras.callback

    Some functions are still not implement yet.
    """
    def __init__(self, early_stopping=False, validation_feed_dict=None, save_iterval=10000,
                 save_path=None, learning_rate_schduler=None, verbose_interval=500):
        self._training_iterations = 0
        self._saving_iterval = save_iterval
        self._saver = None
        self._early_stop = early_stopping
        self._save_path = save_path
        self._learning_rate_schduler = learning_rate_schduler
        self._loss = 0.0
        self._verbose_interval = verbose_interval

    def train(self, loss, training_step, feed_dict, sess):
        _loss_value, __ = \
            sess.run([loss, training_step], feed_dict=feed_dict)

        self._loss += _loss_value
        self._training_iterations += 1
        if self._training_iterations % self._verbose_interval == 0:
            print('Training process at iteration %d, loss %.5f' %
                  (self._training_iterations, self._loss / self._verbose_interval))
            self._loss = 0

        if self._training_iterations % self._saving_iterval == 0:
            print('Saving model at iteration %d towards file %s' %
                  (self._training_iterations, self._save_path))
