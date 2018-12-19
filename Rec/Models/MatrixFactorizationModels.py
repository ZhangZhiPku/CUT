import tensorflow as tf
import pandas as pd
import numpy as np

from Rec.Common.Utilites import Training_Data_Batcher, Training_Helper


class SVD_Rec_Model():

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

        # tensorflow placeholders and variables:
        self._userid_input_placeholder = None
        self._itemid_input_placeholder = None
        self._ratings_input_placeholder = None
        self._user_embedding_lookup_table = None
        self._item_embedding_lookup_table = None
        self._user_bias_lookup_table = None
        self._item_bias_lookup_table = None

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

    def lock_on_dataset(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            distinct_user_ids = dataframe['uid'].unique()
            distinct_item_ids = dataframe['iid'].unique()

            self._max_rating = dataframe['ratings'].max()
            self._min_rating = dataframe['ratings'].min()
            # mapping user id and item id into continuous vector space
            self._user_mapping, self._inverse_user_mapping = \
                {uid: pos for pos, uid in enumerate(distinct_user_ids)}, \
                {pos: uid for pos, uid in enumerate(distinct_user_ids)}
            self._item_mapping, self._inverse_item_mapping = \
                {iid: pos for pos, iid in enumerate(distinct_item_ids)}, \
                {pos: iid for pos, iid in enumerate(distinct_item_ids)}
            self.n_users, self.n_items = len(distinct_user_ids), len(distinct_item_ids)

            self._rating_variance = dataframe['ratings'].std()
            self._rating_mean = dataframe['ratings'].mean()

            self._training_data = dataframe.copy()
            self._training_data['uid'] = dataframe['uid'].apply(lambda x: self._user_mapping[x])
            self._training_data['iid'] = dataframe['iid'].apply(lambda x: self._item_mapping[x])
            self._training_data['ratings'] = (dataframe['ratings'] - self._rating_mean) / self._rating_variance
        else:
            raise Exception(
                'Input variable dataframe must be one of pandas dataframe, but {0} are given'.format(
                    type(dataframe)))

    def build(self, rank, l2_gamma, learning_rate=1e-3):
        with tf.name_scope('SVD-learnable-variables'):
            with tf.device('/cpu:0'):
                self._user_embedding_lookup_table = tf.get_variable(
                    name='SVD-user-features', shape=[self.n_users, rank],
                    initializer=tf.initializers.random_uniform(minval=-0.17, maxval=0.17),
                    trainable=True)
                self._item_embedding_lookup_table = tf.get_variable(
                    name='SVD-item-features', shape=[self.n_items, rank],
                    initializer=tf.initializers.random_uniform(minval=-0.17, maxval=0.17),
                    trainable=True)
                self._user_bias_lookup_table = tf.get_variable(
                    name='SVD-user-bias', shape=[self.n_users],
                    initializer=tf.initializers.random_uniform(minval=-0.17, maxval=0.17),
                    trainable=True)
                self._item_bias_lookup_table = tf.get_variable(
                    name='SVD-item-bias', shape=[self.n_items],
                    initializer=tf.initializers.random_uniform(minval=-0.17, maxval=0.17),
                    trainable=True)

        with tf.name_scope('SVD-input-placeholders'):
            self._userid_input_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None], name='SVD-uid-placeholder')
            self._itemid_input_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None], name='SVD-iid-placeholder')
            self._ratings_input_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[None], name='SVD-rating-placeholder')

        with tf.name_scope('SVD-kernel'):
            user_vectors = tf.nn.embedding_lookup(
                params=self._user_embedding_lookup_table, ids=self._userid_input_placeholder)
            item_vectors = tf.nn.embedding_lookup(
                params=self._item_embedding_lookup_table, ids=self._itemid_input_placeholder)
            user_bias = tf.nn.embedding_lookup(
                params=self._user_bias_lookup_table, ids=self._userid_input_placeholder)
            item_bias = tf.nn.embedding_lookup(
                params=self._item_bias_lookup_table, ids=self._itemid_input_placeholder)

            self._raw_output = tf.multiply(user_vectors, item_vectors)
            self._raw_output = tf.reduce_sum(self._raw_output, axis=1)
            self._raw_output += (user_bias + item_bias)
            self._raw_output = tf.reshape(self._raw_output, shape=[-1])

            self._prediction = self._raw_output * self._rating_variance + self._rating_mean
            self._prediction = tf.clip_by_value(self._prediction,
                                                clip_value_min=self._min_rating, clip_value_max=self._max_rating)

        with tf.name_scope('SVD-loss'):
            self.loss = tf.losses.mean_squared_error(self._raw_output,
                                                     self._ratings_input_placeholder)
            # normalization loss add-on
            self.loss += l2_gamma * tf.reduce_mean(
                tf.pow(user_vectors, 2))
            self.loss += l2_gamma * tf.reduce_mean(
                tf.pow(item_vectors, 2))

        with tf.name_scope('SVD-training-step'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            training_step = optimizer.minimize(self.loss)

        return training_step

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