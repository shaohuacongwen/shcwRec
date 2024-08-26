import tensorflow as tf
from model import *

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

def _user_profile_interest_attention(user_interests, user_profile):
    bs, num_all_interest, user_emb_size = get_shape(user_profile)
    print(bs)
    print(num_all_interest)
    print(user_emb_size)
    user_attention_weights = tf.concat([user_profile, user_interests], axis=-1)  # [None, 6, 40 + 352]
    user_attention_units = [8, 3]

    with tf.variable_scope("user_attention_net", reuse=tf.AUTO_REUSE):
        bias = True
        biases_initializer = tf.zeros_initializer() if bias else None

        for i, units in enumerate(user_attention_units):
            activation_fn = (tf.nn.relu if i < len(user_attention_units) - 1
                             else tf.nn.sigmoid)
            user_attention_weights = layers.fully_connected(
                user_attention_weights, units, activation_fn=None,
                biases_initializer=biases_initializer)
            user_attention_weights = activation_fn(user_attention_weights)

#    print("user_profile shape:", tf.shape(user_profile))
#    print("Expected shape:", [bs, num_all_interest, 3, int(user_emb_size / 3)])
    print("user_profile shape before reshape:", tf.shape(user_profile))

    user_multi_features = tf.reshape(user_profile, [bs, num_all_interest, 3, int(user_emb_size / 3)])  # [None, 6, 5, 8]
    user_attended_features = tf.multiply(user_multi_features,
                                         tf.expand_dims(user_attention_weights, axis=3))  # [None, 6, 5, 1]

    user_attended_profile = tf.reshape(user_attended_features, [bs, num_all_interest, user_emb_size])

    return user_attended_profile

    
class Model_UMI(Model): #用这个model需要将embedding_dim和hidden_size调整成3的倍数，比如66  效果很差
    def __init__(self, n_mid, n_uid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_UMI, self).__init__(n_mid, n_uid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="UMI")

        margin=0.001
        alpha1=alpha2=0.1
        alpha=10
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)
            '''            
        uid_batch_embedded = tf.tile(tf.expand_dims(self.uid_batch_embedded, 1),[1, num_interest, 1]) #维度对齐
        ir_hidden = tf.concat([uid_batch_embedded, interest_emb], axis=-1)
#        self.user_eb = interest_emb + 0.3 * tf.expand_dims(self.uid_batch_embedded, 1)
#        user_interest_emb = interest_emb 

#        self.user_eb = tf.layers.dense(self.user_eb, units=embedding_dim * 2, activation=None)
#        self.user_eb = tf.layers.dense(self.user_eb, units=embedding_dim, activation=tf.nn.relu)
        ir_hidden = tf.layers.dense(ir_hidden, units=embedding_dim // 2, activation=tf.nn.relu)
        ir_hidden = tf.layers.dense(ir_hidden, units=embedding_dim, activation=None)
        
        ir_output = tf.nn.sigmoid(ir_hidden)
        self.user_eb = tf.concat([interest_emb, ir_output], axis=-1)
        self.user_eb = tf.layers.dense(self.user_eb, units=embedding_dim * 2, activation=None)
        self.user_eb = tf.layers.dense(self.user_eb, units=embedding_dim, activation=tf.nn.relu)'''
        self.user_eb = _user_profile_interest_attention(interest_emb, tf.tile(tf.expand_dims(self.uid_batch_embedded, 1),[1, num_interest, 1]))
        self.user_eb = tf.concat([self.user_eb, interest_emb], axis=-1)
        self.user_eb = tf.layers.dense(self.user_eb, units=embedding_dim * 2, activation=None)
        self.user_eb = tf.layers.dense(self.user_eb, units=embedding_dim, activation=tf.nn.relu)

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads)
        
#        readout = tf.nn.l2_normalize(readout, axis=1) #归一化

        self.build_full_softmax_loss(self.item_eb, readout)