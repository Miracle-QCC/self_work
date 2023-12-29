import torch
import numpy as np
from sac_model import SAC
import matplotlib.pyplot as plt
obs_dim = 2
act_dim = 1
act_bound = 1 # 14
seed = 42
sac = SAC(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, seed=seed, )
sac.ac.pi.load_state_dict(torch.load('actor.pth'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datas = []

for i in range(1000):
    x_sample = np.random.rand(128) * 20 - 10
    y_sample = np.random.rand(128) * 20 - 10
    s = np.stack([x_sample,y_sample],axis=1)
    with torch.no_grad():
        a = sac.get_action(s,deterministic=True) * 10 - 5
        data = np.concatenate([x_sample.reshape(-1,1),y_sample.reshape(-1,1),a],axis=-1)
        datas.append(data)

datas = np.concatenate(datas)
unique_data, counts = np.unique(data, axis=0, return_counts=True)
frequency = counts / len(datas)

# 设置颜色映射
color_map = plt.cm.get_cmap('viridis')

# 绘制散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(unique_data[:, 0], unique_data[:, 1], unique_data[:, 2], c=frequency, cmap=color_map, marker='o')

# 设置颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Frequency')

# 设置坐标轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.title('3D Scatter Plot with Frequency-based Color')
plt.show()