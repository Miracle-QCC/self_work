import torch
import numpy as np
import normflows as nf

from sklearn.datasets import make_moons

from matplotlib import pyplot as plt

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
nfm = nf.NormalizingFlow(q0=q0, flows=flows)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)

# Train model
max_iter = 1000
show_iter = 500
batch_size = 1024
loss_hist = np.array([])
data = np.load('my_data.npy')
indices = np.random.choice(data.shape[0], batch_size, replace=False)

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    indices = np.random.choice(data.shape[0], batch_size, replace=False)

    x = torch.from_numpy(data[indices]).float().to(device)

    # Compute loss
    loss = nfm.forward_kld(x)

    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    # # Plot learned distribution
    # if (it + 1) % show_iter == 0:
    #     nfm.eval()
    #     log_prob = nfm.log_prob(zz)
    #     nfm.train()
    #     prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    #     prob[torch.isnan(prob)] = 0
    #
    #     plt.figure(figsize=(2, 2))
    #     plt.pcolormesh(xx, yy, prob.data.numpy())
    #     plt.gca().set_aspect('equal', 'box')
    #     plt.show()

torch.save(nfm.state_dict(), 'flow_model.pth')
# Plot loss
plt.figure(figsize=(2, 2))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()# Plot loss
plt.figure(figsize=(2, 2))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()