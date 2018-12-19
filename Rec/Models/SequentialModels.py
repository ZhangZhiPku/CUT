import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
import math

from Rec.Common.Utilites import Training_Data_Batcher, Training_Helper


class UPS_Cliassifler():
    """
    Classifier with user profile based on sequential data
    """

    def __init__(self):
        self.batcher = None
        self.sess = None

        # data set properties
        self.n_items = 0
        self.n_users = 0
        self._max_rating = 0
        self._min_rating = 0
        self._training_data = None
        self._rating_variance = 0.0
        self._rating_mean = 0.0

        # tensorflow trianing step and loss function:
        self.training_step = None
        self.loss = None
        self._raw_output = None
        self._prediction = None

        # build-in properties
        self._built = False
        self._user_mapping = None
        self._item_mapping = None
        self._inverse_user_mapping = None
        self._inverse_item_mapping = None

    def lock_on_dataset(self, sequential_dataframe, user_tag_dataframe):
        if isinstance(sequential_dataframe, pd.DataFrame) \
                and isinstance(user_tag_dataframe, pd.DataFrame):
            distinct_user_ids = sequential_dataframe['uid'].unique()
            distinct_item_ids = sequential_dataframe['iid'].unique()

            # mapping user id and item id into continuous vector space
            self._user_mapping, self._inverse_user_mapping = \
                {uid: pos for pos, uid in enumerate(distinct_user_ids)}, \
                {pos: uid for pos, uid in enumerate(distinct_user_ids)}
            self._item_mapping, self._inverse_item_mapping = \
                {iid: pos for pos, iid in enumerate(distinct_item_ids)}, \
                {pos: iid for pos, iid in enumerate(distinct_item_ids)}
            self.n_users, self.n_items = len(distinct_user_ids), len(distinct_item_ids)
        else:
            raise Exception(
                'Input variable dataframe must be one of pandas dataframe, but {0} are given'.format(
                    type(sequential_dataframe) + type(user_tag_dataframe)))

    def build(self, rank, learning_rate=1e-3, seq_length=100, classes=2, n_neg_samplings=32):

        with tf.name_scope('embeddings'):
            with tf.device('/cpu:0'):
                self._item_embedding_lookup_table = tf.get_variable(
                    name='item-embeddings', shape=[self.n_items, rank],
                    initializer=tf.initializers.random_uniform(minval=-0.17, maxval=0.17),
                    trainable=True)

        with tf.name_scope('nce-sampling-variables'):
            with tf.device('/cpu:0'):
                self._nce_weights = tf.get_variable(
                    name='nce-weights', shape=[self.n_items, rank],
                    initializer=tf.initializers.truncated_normal(stddev=1.0 / math.sqrt(rank)))
                self._nce_biases = tf.Variable(tf.zeros([self.n_items]))

        with tf.name_scope('pretrain-input-placeholders'):
            self._sequential_window_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None], name='sequential-window-input')
            self._sequential_label_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name='sequential-label-input')

        with tf.name_scope('pretrain-kernel'):
            _sequential_window_embeddings = tf.nn.embedding_lookup(params=self._item_embedding_lookup_table,
                                                                   ids=self._sequential_window_placeholder)

        with tf.name_scope('finetune-input-placeholders'):
            self._user_trajoctry_input = tf.placeholder(
                dtype=tf.int32, shape=[None, seq_length], name='sequential-window-input')
            self._user_tags = tf.placeholder(
                dtype=tf.int32, shape=[None, classes], name='sequential-label-input')

        with tf.name_scope('finetune-kernel'):
            user_trajectory_embeddings = tf.nn.embedding_lookup(
                params=self._item_embedding_lookup_table, ids=self._user_trajoctry_input)

            user_embeddings = Bidirectional(CuDNNGRU(units=128))(user_trajectory_embeddings)
            self._raw_output = Dense(units=1, activation='sigmoid')(user_embeddings)
            self._prediction = tf.argmax(self._raw_output, axis=1)

        with tf.name_scope('losses'):
            self.pretraining_loss = tf.nn.nce_loss(
                weights=self._nce_weights,
                biases=self._nce_biases,
                labels=self._sequential_label_placeholder,
                inputs=_sequential_window_embeddings,
                num_sampled=n_neg_samplings,
                num_classes=self.n_items
            )
            self.finetune_loss = tf.losses.softmax_cross_entropy(onehot_labels=self._user_tags,
                                                                 logits=self._raw_output)

        with tf.name_scope('training-step'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            training_step = optimizer.minimize(self.loss)

        return training_step

    def pretrain(self):
        pass

    def finetune(self):
        pass

    def fit(self, dataframe, learning_rate=1e-3, iteration=17000, batchsize=1024,
            l2_gamma=0.3, rank=4, verbose=1):

        self.lock_on_dataset(dataframe)

        tf.reset_default_graph()
        training_step = self.build(rank, l2_gamma, learning_rate)
        training_agent = Training_Helper()
        batcher = Training_Data_Batcher(self._training_data, batch_size=batchsize)

        self.sess = tf.Session()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        for iteration_step in range(iteration):
            batch_data = batcher.make_batch()
            training_agent.train(self.loss, training_step, feed_dict={
                self._userid_input_placeholder: batch_data['uid'],
                self._itemid_input_placeholder: batch_data['iid'],
                self._ratings_input_placeholder: batch_data['ratings']
            }, sess=self.sess)

    def predict(self, dataframe, batchsize=1024, verbose=1):
        _data = pd.DataFrame()
        _data['uid'] = dataframe['uid'].apply(lambda x: self._user_mapping[x])
        _data['iid'] = dataframe['iid'].apply(lambda x: self._item_mapping[x])

        batcher = Training_Data_Batcher(_data, batch_size=batchsize, reshuffle=False)
        predictions = []
        for batch_data in batcher.make_walk_through():
            predictions.append(self.sess.run(
                self._prediction, feed_dict={
                    self._userid_input_placeholder: batch_data['uid'],
                    self._itemid_input_placeholder: batch_data['iid']
                }
            ))
        return np.concatenate(predictions, axis=0)