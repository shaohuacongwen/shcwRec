import tensorflow as tf
from model import *

class Model_DNN(Model):
    def __init__(self, n_mid, n_uid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, n_uid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)