import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tqdm import tqdm
K = 64
torch.manual_seed(0)

latent_size = 3
hidden_units = 256
hidden_layers = 3

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set prior and q0
q0 = nf.distributions.DiagGaussian(3, trainable=False)

# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows).to(device)

nfm.load_state_dict(torch.load('flow_model.pth'))


grid_size = 1000
xx, yy, aa = torch.meshgrid(torch.linspace(-10, 10, grid_size), torch.linspace(-10, 10, grid_size), torch.linspace(-5, 5, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2), aa.unsqueeze(2)], 2).view(-1, 3)

# Plot learned distribution
nfm.eval()
n = len(zz)
log_probs = []
with torch.no_grad():
    for i in range(n // 1000):
        log_probs.append(nfm.log_prob(zz[i*1000:(i+1)*1000].to(device)).to(device).view(-1,1))
# nfm.train()
log_probs = torch.stack(log_probs)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用 scatter 绘制四维散点图，颜色映射到 prob
scatter = ax.scatter(xx, yy, aa, c=prob.detach().cpu().numpy(), cmap='viridis', marker='o')

# 添加颜色条
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()