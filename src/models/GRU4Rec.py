import tensorflow as tf
from model import *

class Model_GRU4REC(Model):
    def __init__(self, n_mid, n_uid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_GRU4REC, self).__init__(n_mid, n_uid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = self.mask_length
            rnn_outputs, final_state1 = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")

        self.user_eb = final_state1
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)