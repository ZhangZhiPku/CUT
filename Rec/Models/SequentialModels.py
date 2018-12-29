import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import skipgrams
import pandas as pd
import numpy as np
import math

from Rec.Common.Utilites import Training_Data_Batcher, Training_Helper


class UPS_Cliassifler():
    """
    Classifier with user profile based on sequential data
    """

    def __init__(self, n_items):
        self.batcher = None
        self.sess = None

        # data set properties
        self.n_items = n_items

        # tensorflow trianing step and loss function:
        self.training_step = None
        self.loss = None
        self._raw_output = None
        self._prediction = None

        # build-in properties
        self._built = False

    def build(self, rank, learning_rate=1e-3, seq_length=100, classes=2, n_neg_samplings=64):

        with tf.name_scope('embeddings'):
            with tf.device('/cpu:0'):
                self._item_embedding_lookup_table = tf.get_variable(
                    name='item-embeddings', shape=[self.n_items, rank],
                    initializer=tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
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
                dtype=tf.int32, shape=[None, 1], name='sequential-label-input')

        with tf.name_scope('finetune-kernel'):
            user_trajectory_embeddings = tf.nn.embedding_lookup(
                params=self._item_embedding_lookup_table, ids=self._user_trajoctry_input)

            user_embeddings = Bidirectional(CuDNNGRU(units=128))(user_trajectory_embeddings)
            self._raw_output = Dense(units=1, activation='sigmoid')(user_embeddings)
            self._prediction = tf.argmax(self._raw_output, axis=1)

        with tf.name_scope('losses'):
            self.pretraining_loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=self._nce_weights,
                biases=self._nce_biases,
                labels=self._sequential_label_placeholder,
                inputs=_sequential_window_embeddings,
                num_sampled=n_neg_samplings,
                num_classes=self.n_items
            ))
            self.finetune_loss = tf.losses.softmax_cross_entropy(onehot_labels=self._user_tags,
                                                                 logits=self._raw_output)

        with tf.name_scope('training-step'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.pretrain_step = optimizer.minimize(self.pretraining_loss)

    def pretrain(self, sequential_input, learning_rate=1e-3, iteration=800000, batchsize=512):
        sentences = pd.Series([trajectory for uid, trajectory in sequential_input])
        pretrain_couples = []
        for sentence in sentences:
            _couple, _label = skipgrams(sentence, window_size=3,
                                        negative_samples=0, vocabulary_size=self.n_items)
            pretrain_couples.extend(_couple)
        pretrain_dataframe = pd.DataFrame(pretrain_couples, columns=['relative_word', 'target_word'])

        print(pretrain_dataframe)

        self.build(rank=16)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        batcher = Training_Data_Batcher(pretrain_dataframe, batch_size=batchsize)
        training_agent = Training_Helper()

        for iteration_step in range(iteration):
            training_batch = batcher.make_batch()
            training_agent.train(
                self.pretraining_loss, training_step=self.pretraining_loss,
                feed_dict={
                    self._sequential_label_placeholder: np.reshape(training_batch['target_word'].values, [-1, 1]),
                    self._sequential_window_placeholder: training_batch['relative_word']},
                sess=self.sess
            )

        sentences_lengths = sentences.apply(lambda x: len(x))
        cut_length = sentences_lengths.quantile(q=0.95)
        print('sequence cut length has been set as %d' % cut_length)

    def finetune(self):
        pass

    def predict(self, dataframe, batchsize=1024, verbose=1):
        pass