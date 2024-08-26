#coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
import fwr13y.d9m.tensorflow as tf_determinism

import numpy as np

import faiss
import tensorflow as tf
from data_iterator import DataIterator
from model import *
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.ComiRec_SA import Model_ComiRec_SA
from models.DNN import Model_DNN
from models.GRU4Rec import Model_GRU4REC
from models.MIND import Model_MIND
from models.UMI import Model_UMI
from models.cx import Model_cx

    
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=1240)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)

differ = []
best_metric = 0
item_embs_history = []

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def compute_diversity(item_list, item_cate_map, default_cate='default'):
    n = len(item_list)
    diversity = 0.0
    valid_pairs = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if item_list[i] in item_cate_map and item_list[j] in item_cate_map:
                diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
                valid_pairs += 1
            else:
                continue
#                print(f"Missing item in item_cate_map: {item_list[i] if item_list[i] not in item_cate_map else item_list[j]}")
    
    if valid_pairs > 0:
        diversity /= valid_pairs
    else:
        diversity = 0.0  # 或者设置为某个默认值
    
    return diversity


def evaluate_full(sess, test_data, model, model_path, batch_size, item_cate_map, save=True, coef=None):
#    global item_embs_history
    topN = args.topN

    item_embs = model.output_item(sess)
#    print(item_embs)
#    if len(item_embs_history)!=0:
#        differ.append(np.sum(np.abs(item_embs_history-item_embs)))
#        print("differ=", differ[-1])
#    item_embs_history = item_embs
    
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.output_user(sess, [hist_item, hist_mask, nick_id])

        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            item_cor_list.append(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        item_cor_list.append(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(item_id)
    
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

def get_model(dataset, model_type, item_count, user_count, batch_size, maxlen, args):
    if model_type == 'DNN': 
        model = Model_DNN(item_count, 0, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'GRU4REC': 
        model = Model_GRU4REC(item_count, 0, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(item_count, 0, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, 0, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, user_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, True)
    elif model_type == 'cx':
        model = Model_cx(item_count, user_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, True, args)
    elif model_type == 'UMI':
        model = Model_UMI(item_count, user_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    return model

def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extr_name = current_time
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(args.embedding_dim), 'len'+str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('save_model/runs/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('save_model/runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name

def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        user_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        lr = 0.001,
        max_iter = 100,
        patience = 20,
        args=None
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)

    best_model_path = "save_model/best_model/" + exp_name + '/'

    gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('save_model/runs/' + exp_name)

    item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
        valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)
        
        model = get_model(dataset, model_type, item_count, user_count, batch_size, maxlen, args)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        try:
            loss_sum = 0.0
            trials = 0

            for src, tgt in train_data:
                data_iter = prepare_data(src, tgt)
                loss = model.train(sess, list(data_iter) + [lr])
                
                loss_sum += loss
                iter += 1

                if iter % test_iter == 0:
                    metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map)
                    log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                    print(exp_name)
                    print(log_str)

                    writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                    if metrics != {}:
                        for key, value in metrics.items():
                            writer.add_scalar('eval/' + key, value, iter)
                    
                    if 'recall' in metrics:
                        recall = metrics['recall']
                        global best_metric
                        if recall > best_metric:
                            best_metric = recall
                            model.save(sess, best_model_path)
                            trials = 0
                        else:
                            trials += 1
                            if trials > patience:
                                break

                    loss_sum = 0.0
                    test_time = time.time()
                    print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                    sys.stdout.flush()
                
                if iter >= max_iter * 1000:
                        break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # 在训练结束后保存最终结果和parser参数
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        model.restore(sess, best_model_path)
        
  
        result_path = "results/"
        
        # 保存parser参数到文件
        parser_params_file = result_path + exp_name + '_result.txt'
        final_metrics_file = result_path + exp_name + '_result.txt'
        result = exp_name + '\n'
        with open(parser_params_file, 'w') as f:
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)}\n')
                result += f'{arg}: {getattr(args, arg)}\n'
            
            final_metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map, save=False)
            print(', '.join(['valid ' + key + ': %.6f' % value for key, value in final_metrics.items()]))
            final_metrics_str = ', '.join(['final ' + key + ': %.6f' % value for key, value in final_metrics.items()])
            f.write(final_metrics_str + '\n')
            result += final_metrics_str + '\n'

            test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
            test_metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, item_cate_map, save=False)
            print(', '.join(['test ' + key + ': %.6f' % value for key, value in test_metrics.items()]))
            test_metrics_str = ', '.join(['test ' + key + ': %.6f' % value for key, value in test_metrics.items()])
            f.write(test_metrics_str + '\n')
            result += test_metrics_str + '\n'
#            print(test_metrics_str)
            
#        with open('result.txt','a') as f:
#            f.write(result+'\n')

#        differ_file = reslt_path + 'differ.txt'
#        with open(differ_file, 'w') as f:
#            f.write(str(differ))
                    
        
#        item_embs = model.output_item(sess)
#        np.save('output/item_embs.npy', item_embs)
'''        # t-SNE 可视化
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        item_embs_2d = tsne.fit_transform(item_embs)

        # 绘制 t-SNE 聚类图
        plt.figure(figsize=(10, 8))
        plt.scatter(item_embs_2d[:, 0], item_embs_2d[:, 1], s=5, c='blue', alpha=0.6)
        plt.title('t-SNE Visualization of Item Embeddings')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
#        plt.show()
        plt.savefig('output/' + exp_name + '_tsne_visualization.png', format='png', dpi=300)
        plt.show()'''
'''
        # 计算每个物品的交互频率
        item_frequencies = defaultdict(int)
        with open('path/to/train_data.txt', 'r') as f:
            for line in f:
                _, item_id, _ = line.strip().split(',')
                item_frequencies[int(item_id)] += 1

        # 将交互频率映射到颜色上
        max_frequency = max(item_frequencies.values())
        normalized_frequencies = np.array([item_frequencies[i] / max_frequency for i in range(item_count)])

        # t-SNE 降维可视化
        tsne = TSNE(n_components=2, random_state=42)
        item_embs_2d = tsne.fit_transform(item_embs)

        # 使用颜色映射来区分交互频率
        cmap = cm.get_cmap('coolwarm')
        colors = cmap(normalized_frequencies)

        # 绘制 t-SNE 聚类图
        plt.figure(figsize=(10, 8))
        plt.scatter(item_embs_2d[:, 0], item_embs_2d[:, 1], s=5, c=colors, alpha=0.6)
        plt.title('t-SNE Visualization of Item Embeddings with Interaction Frequencies')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # 保存图像
        plt.savefig('output/' + exp_name + '_tsne_visualization_colored.png', format='png', dpi=300)
        plt.show()'''
        
def test(
        test_file,
        cate_file,
        item_count,
        user_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "save_model/best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, user_count, batch_size, maxlen)
    item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, item_cate_map, save=False, coef=args.coef)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        item_count,
        user_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "save_model/best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, user_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)
        
        
if __name__ == '__main__':

    print(sys.argv)
    args = parser.parse_args()
    
    SEED = args.random_seed
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    tf_determinism.enable_determinism()
    
    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    elif args.dataset == 'rocket':
        batch_size = 256
        seq_len = 20
        maxlen = 20
        test_iter = 200
        path = './data/rocket_data/'
        item_count = 81635 + 1
        user_count = 33708 + 1
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file, 
              item_count=item_count, user_count = user_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter, 
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience, args=args)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, user_count = user_count, dataset=dataset, batch_size=batch_size, 
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, user_count = user_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')
'''
    # 在训练结束后获取embedding并生成tsne图
    model.restore(sess, best_model_path)
    item_embs = model.output_item(sess)  # 获取物品嵌入向量

    # 创建标签或可以根据需要自定义分类
    labels = [i % 10 for i in range(item_embs.shape[0])]  # 简单地创建一个标签用于可视化
    plot_tsne(item_embs, labels, save_path='item_embeddings_tsne.png')
    '''