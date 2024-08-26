import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers
import fwr13y.d9m.tensorflow as tf_determinism
#SEED = args.random_seed
#tf.set_random_seed(SEED)
#np.random.seed(SEED)
#tf_determinism.enable_determinism()

class Model(object):
    def __init__(self, n_mid, n_uid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # 定义嵌入层的命名空间
        with tf.name_scope('Embedding_layer'):
            
            #物品嵌入
            # 创建一个变量用于存储物品的嵌入向量
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            # 创建一个变量用于存储物品的偏置值，初始化为0，不可训练
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(), trainable=False)
            # 根据物品ID查找对应的嵌入向量
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            # 根据物品历史ID查找对应的嵌入向量
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            
            #用户嵌入
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, embedding_dim], trainable=True)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)


        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1,seq_len, 1))
        '''        # 归一化当前物品的嵌入向量
        self.item_eb = tf.nn.l2_normalize(self.mid_batch_embedded, axis=1)
        # 将历史物品的嵌入向量乘以mask后进行归一化
        self.item_his_eb = tf.nn.l2_normalize(self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1)), axis=2)'''
    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_full_softmax_loss(self, item_emb, user_emb):
        # 计算 logits
        logits = tf.matmul(user_emb, tf.transpose(self.mid_embeddings_var)) + self.mid_embeddings_bias

        # 计算 softmax 概率分布
        softmax_probs = tf.nn.softmax(logits)

        # 获取标签
        labels = tf.reshape(self.mid_batch_ph, [-1])

        # 创建一个one-hot编码的标签
        one_hot_labels = tf.one_hot(labels, depth=logits.shape[1])
        # 计算交叉熵损失
        cross_entropy_loss = -tf.reduce_sum(one_hot_labels * tf.log(softmax_probs + 1e-9), axis=1)

        # 计算批次的平均损失
        self.loss = tf.reduce_mean(cross_entropy_loss)

        # 使用 Adam 优化器来最小化损失
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
    def build_qfl_softmax_loss(self, item_emb, user_emb):
        alpha = 0.99
        gamma = 2

        # 计算 logits
        logits = tf.matmul(user_emb, tf.transpose(self.mid_embeddings_var)) + self.mid_embeddings_bias

        # 计算 softmax 概率分布
        softmax_probs = tf.nn.softmax(logits)

        # 获取标签并创建 one-hot 编码的标签
        labels = tf.reshape(self.mid_batch_ph, [-1])
        one_hot_labels = tf.one_hot(labels, depth=logits.shape[1])

        # 计算 pt
        pt = tf.reduce_sum(one_hot_labels * softmax_probs, axis=1)

        # 计算QFL的权重
        alpha_t = one_hot_labels * alpha + (1 - one_hot_labels) * (1 - alpha)
        weight = alpha_t * tf.expand_dims(tf.pow((1 - pt), gamma), axis=-1)

        # 计算加权的交叉熵损失
        cross_entropy_loss = -tf.reduce_sum(weight * tf.log(softmax_probs + 1e-9), axis=1)

        # 计算批次的平均损失
        self.loss = tf.reduce_mean(cross_entropy_loss)

        # 使用 Adam 优化器来最小化损失
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def build_contrastive_loss(self, user_emb, item_emb, labels):
        # 计算正样本距离
        distances = tf.reduce_sum(tf.square(user_emb - item_emb), axis=-1)

        # 对比损失
        pos_loss = labels * tf.square(distances)
        neg_loss = (1 - labels) * tf.square(tf.maximum(self.margin - tf.sqrt(distances + 1e-6), 0))

        loss = tf.reduce_mean(0.5 * (pos_loss + neg_loss))
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_covariance_loss(self):#尝试正则化一下
        X = tf.concat([self.mid_embeddings_var], axis=0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X -= tf.reduce_mean(X, axis=0)
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        cov_loss = tf.reduce_sum(tf.square(tf.linalg.set_diag(cov, tf.zeros([tf.shape(cov)[0]], dtype=tf.float32))))
        return 0.1 * cov_loss
        
    def build_triplet_loss(self, item_emb, user_emb):
        # 正样本距离
        pos_dist = tf.reduce_sum(tf.square(user_emb - item_emb), axis=-1)

        # 负样本随机采样
        neg_samples = tf.random_uniform([self.batch_size, self.neg_num], maxval=self.n_mid, dtype=tf.int32)
        neg_samples_emb = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples)
        
        # 负样本距离
        neg_distances = tf.reduce_sum(tf.square(tf.expand_dims(user_emb, 1) - neg_samples_emb), axis=-1)
        neg_dist = tf.reduce_min(neg_distances, axis=-1)

        # 计算三元组损失
        basic_loss = pos_dist - neg_dist + self.margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_quadruplet_loss(self, item_emb, user_emb):
        
        # 正样本距离
        pos_dist = tf.reduce_sum(tf.square(user_emb - item_emb), axis=-1)

        # 随机采样负样本
        neg_samples = tf.random_uniform([self.batch_size, self.neg_num], maxval=self.n_mid, dtype=tf.int32)
        neg_samples_emb = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples)
        
        # 负样本距离
        neg_distances = tf.reduce_sum(tf.square(tf.expand_dims(user_emb, 1) - neg_samples_emb), axis=-1)
        neg_dist1 = tf.reduce_min(neg_distances, axis=-1)

        # 再次随机采样负样本
        neg_samples2 = tf.random_uniform([self.batch_size, self.neg_num], maxval=self.n_mid, dtype=tf.int32)
        neg_samples_emb2 = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples2)

        # 负样本距离2
        neg_distances2 = tf.reduce_sum(tf.square(tf.expand_dims(user_emb, 1) - neg_samples_emb2), axis=-1)
        neg_dist2 = tf.reduce_min(neg_distances2, axis=-1)

        # 计算四元组损失
        loss1 = tf.reduce_sum(tf.maximum(pos_dist - neg_dist1 + self.alpha1, 0.0))
        loss2 = tf.reduce_sum(tf.maximum(pos_dist - neg_dist2 + self.alpha2, 0.0))

        qloss = loss1 + loss2
        
        cov_loss = self.build_covariance_loss()
        loss = qloss + cov_loss
        
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_structured_loss(self, item_emb, user_emb):
        # 计算距离
        pos_dist = tf.reduce_sum(tf.square(user_emb - item_emb), axis=-1)

        # 随机采样负样本
        neg_samples = tf.random_uniform([self.batch_size, self.neg_num], maxval=self.n_mid, dtype=tf.int32)
        neg_samples_emb = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples)

        # 计算负样本距离
        neg_distances = tf.reduce_sum(tf.square(tf.expand_dims(user_emb, 1) - neg_samples_emb), axis=-1)

        # 计算 Structured Loss
        max_neg_dist = tf.reduce_max(self.alpha - neg_distances, axis=-1)
        structured_loss = tf.reduce_sum(tf.maximum(0.0, max_neg_dist + pos_dist))

        # 协方差正则化
        cov_loss = self.build_covariance_loss()

        # 总损失
        loss = structured_loss + cov_loss

        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_loss(self, item_emb, user_emb, labels):
        # Softmax Loss
        with tf.variable_scope("softmax"):
            logits = tf.layers.dense(user_emb, self.num_classes, activation=None)
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Center Loss
        center_loss = self.build_center_loss(item_emb, labels)

        # Total Loss
        total_loss = softmax_loss + self.center_loss_weight * center_loss
        self.loss = total_loss

        # Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.center_update_op)

    def build_CPE_loss(self, item_emb, user_emb):
        # 正样本距离
        pos_distance = tf.reduce_sum(tf.square(user_emb - item_emb), axis=-1, name="pos_distance")

        # 随机采样负样本
        neg_samples = tf.random_uniform([self.batch_size, self.neg_num], maxval=self.n_mid, dtype=tf.int32)
        neg_samples_emb = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples)

        # 负样本距离
        neg_distance = tf.reduce_sum(tf.square(tf.expand_dims(user_emb, 1) - neg_samples_emb), axis=-1, name="neg_distance")

        # 损失计算
        gamma_s = neg_distance - tf.expand_dims(pos_distance, -1)
        embedding_loss = tf.reduce_sum(tf.maximum(gamma_s - self.margin, 0) + tf.maximum(self.margin - gamma_s, 0))

        # 协方差损失
        if self.reg:
            X = tf.concat([self.mid_embeddings_var, self.mid_embeddings_bias], axis=0)
            n_rows = tf.cast(tf.shape(X)[0], tf.float32)
            X -= tf.reduce_mean(X, axis=0)
            cov = tf.matmul(X, X, transpose_a=True) / n_rows

            # 使用Cholesky分解计算对数行列式
            L = tf.cholesky(cov)
            log_determinant_of_cov = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))

            cov_loss = (tf.trace(cov) - log_determinant_of_cov) * self.alpha
            loss = embedding_loss + cov_loss
        else:
            loss = embedding_loss

        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        
    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1],
            self.uid_batch_ph: inps[2]  # 添加这一行，确保传递 uid_batch_ph
        })
        return user_embs
    
    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

    
def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape