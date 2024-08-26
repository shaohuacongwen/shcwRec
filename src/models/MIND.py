import tensorflow as tf
from model import *



class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout

class Model_MIND(Model):
    def __init__(self, n_mid, n_uid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True, relu_layer=True):
        super(Model_MIND, self).__init__(n_mid, n_uid, embedding_dim, hidden_size, batch_size, seq_len, flag="MIND")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=0, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_softmax_loss(self.item_eb, self.readout)