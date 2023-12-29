from sac_model import SAC
import torch
import numpy as np
import normflows as nf
from tqdm import tqdm
K = 64
torch.manual_seed(0)
A = np.array([[0.8,0.5],[-0.4,1.2]])
B = np.array([0,1])
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

nfm.load_state_dict(torch.load('flow_model.pth'))
obs_dim = 2
act_dim = 1
act_bound = 1 # 14
seed = 42
sac = SAC(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound, seed=seed, )
batch_size = 256
def dp_process(x,u):
    s = []
    for i in range(128):
        s.append(np.dot(A,x[i]) + B * u[i])
    s = np.array(s)
    return s

if __name__ == '__main__':
    # get data
    MAX_STEPS = int(1e6)
    for i in range(1000):
        np.random.seed(i+1)
        x_sample = np.random.rand(128) * 20 - 10
        y_sample = np.random.rand(128) * 20 - 10
        a = (np.random.rand(128) * 2 - 1).reshape(-1,act_dim)

        s0 = np.stack([x_sample,y_sample],axis=-1)

        s1 = dp_process(s0, a)
        with torch.no_grad():
            r = nfm.log_prob(torch.cat([torch.from_numpy(s0),torch.from_numpy(a)],dim=-1).float().to(device))

        sac.store_batch(s0,a,r.cpu().numpy(),s1)

        # if len(sac.replay_buffer) > 10 * batch_size:
        #
        #     sac.update(batch_size=256)
    for e in tqdm(range(MAX_STEPS)):
        sac.update(batch_size=256)

    torch.save(sac.ac.pi.state_dict(), 'actor.pth')
    torch.save(sac.ac.q1.state_dict(), 'q1.pth')
    torch.save(sac.ac.q2.state_dict(), 'q2.pth')




