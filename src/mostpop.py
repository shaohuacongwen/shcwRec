import numpy as np
from collections import defaultdict

class MostPopular:
    def __init__(self, source):
        # 读取训练数据
        self.train_graph = self.read('data/' + source + '_data/' + source + '_train.txt')

    def read(self, source):
        graph = {}
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                if user_id not in graph:
                    graph[user_id] = []
                graph[user_id].append(item_id)
        return graph
    
    def calculate_item_frequencies(self):
        # 统计每个物品的交互频率
        item_count = defaultdict(int)
        for user in self.train_graph.keys():
            for item in self.train_graph[user]:
                item_count[item] += 1
        return item_count

# 使用示例
def save_item_frequencies(source):
    data = MostPopular(source)
    item_frequencies = data.calculate_item_frequencies()

    # 将 item_frequencies 转换为数组并保存
    max_item_id = max(item_frequencies.keys())  # 找到最大物品ID以便创建数组
    frequency_array = np.zeros(max_item_id + 1)  # 初始化频率数组，长度为最大物品ID+1

    for item_id, freq in item_frequencies.items():
        frequency_array[item_id] = freq

    # 保存为 .npy 文件
    np.save('item_frequencies.npy', frequency_array)
    print(f"Item frequencies saved to 'item_frequencies.npy'")

# 运行此函数进行频率计算并保存
save_item_frequencies('rocket')  # 修改为您的数据路径
