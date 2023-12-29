import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

A = np.array([[0.8,0.5],[-0.4,1.2]])
B = np.array([0,1])
K = np.array([[0.2317,-1.1988]])
# dt=0.0001
# I = np.eye(2)
# F = np.exp(A*dt)
# G = np.dot(np.linalg.inv(A) * (F-I),B)

# def dp_process(s, at):
#     s = np.dot(F,s) + G * at
#     return s
#
# def get_act(s):
#     return -np.dot(K, s)

def dp_process(x,u):
    s = np.dot(A,x) + B * u
    return s

def get_act(x):
    return np.dot(K, x)

# a = np.array([1,1])
S_track = []

def get_data():
    max_a = -10
    min_a = 10
    for i in tqdm(range(100000)):
        s = np.array([np.random.uniform() * 20 - 10 , np.random.uniform() * 20 - 10])
        while True:
            a = get_act(s)
            max_a = max(a, max_a)
            min_a = min(a,min_a)
            S_track.append(s.tolist() + a.tolist())
            s_pre = s
            s = dp_process(s,a)
            dis = np.linalg.norm(s_pre - s)

            # print(dis)
            if dis < 0.0001:
                break
    return max_a, min_a
    # # 将二维点列表拆分成 x 和 y 坐标
    # x_coords, y_coords = zip(*S_track)

max_a, min_a = get_data()
print(min_a)
print(max_a)
# 将数据转换为NumPy数组以便操作
data_array = np.array(S_track)
np.save('my_data.npy', data_array)

# # 提取x、y、z坐标
# x_coords, y_coords, z_coords = data_array[:, 0], data_array[:, 1], data_array[:, 2]
#
# # 创建3D散点图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 使用不同颜色表示数据点的频次
# sc = ax.scatter(x_coords, y_coords, z_coords, c=np.arange(len(data_array)), cmap='viridis')
#
# # 添加颜色条
# cbar = plt.colorbar(sc, ax=ax, label='freq')
#
# # 设置坐标轴标签
# ax.set_xlabel('s1')
# ax.set_ylabel('s2')
# ax.set_zlabel('a')
#
# plt.show(block=True)
# # # 画出轨迹

# plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
# plt.plot(x_coords1, y_coords1, marker='o', linestyle='--', color='g')
#
# plt.title('轨迹')
# plt.xlabel('X坐标')
# plt.ylabel('Y坐标')
# plt.grid(True)
# plt.show()
#
