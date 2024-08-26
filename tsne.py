import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime

# 加载保存的 item_embs 文件
item_embs = np.load('output/item_embs.npy')

# 计算每个物品的交互频率（假设您有频率数据）
item_frequencies = np.load('item_frequencies.npy')  # 加载频率

# 过滤掉交互次数为 0 的物品
mask = item_frequencies > 0
filtered_item_embs = item_embs[mask]
filtered_item_frequencies = item_frequencies[mask]

# 为交互频率大于 0 的物品设置颜色（蓝色），否则为红色
colors = np.array(['blue' if freq > 2 else 'red' for freq in filtered_item_frequencies])

# t-SNE 降维，仅对过滤后的嵌入进行降维
tsne = TSNE(n_components=2, random_state=42)
item_embs_2d = tsne.fit_transform(filtered_item_embs)

# 绘制 t-SNE 聚类图
plt.figure(figsize=(10, 8))
plt.scatter(item_embs_2d[:, 0], item_embs_2d[:, 1], s=5, c=colors, alpha=0.6)
plt.title('t-SNE Visualization of Item Embeddings with Interaction Frequencies (Filtered)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# 获取当前时间作为文件名的一部分
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 保存图像
plt.savefig('output/' + current_time + '.png', format='png', dpi=300)
plt.show()
