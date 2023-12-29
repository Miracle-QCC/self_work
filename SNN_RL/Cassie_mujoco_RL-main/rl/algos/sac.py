import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import sys
import torch
import numpy as np
import gym
import math
from copy import deepcopy
import itertools
from rl.policies.replay_buffer_norm import ReplayBuffer
from rl.policies.popsan import SquashedGaussianPopSpikeActor
from rl.policies.core_cuda import MLPQFunction
os.environ['MUJOCO_KEY_PATH'] = '/home/qcj/.mujoco/mjkey.txt'
from util.env import env_factory


class SpikeActorDeepCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, device,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        try:
            act_limit = action_space.high[0]
        except:
            act_limit = 1
        # build policy and value functions
        self.popsan = SquashedGaussianPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                                    mean_range, std, spike_ts, act_limit, device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, batch_size, deterministic=False):
        with torch.no_grad():
            a, _ = self.popsan(obs, batch_size, deterministic, False)
            a = a.to('cpu')
            return a.numpy()
def spike_sac(env_fn, actor_critic=SpikeActorDeepCritic, ac_kwargs=dict(), seed=0,
              steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, popsan_lr=1e-4, q_lr=1e-3, alpha=0.2, batch_size=100, start_steps=1000,
              update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
              save_freq=10000, norm_clip_limit=3, norm_update=50, tb_comment='', model_idx=0, use_cuda=True):
    """
    Spike Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``popsan`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``popsan`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``popsan`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        popsan_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        norm_clip_limit (float): Clip limit for normalize observation

        norm_update (int): Number of steps to update running mean and var in memory

        tb_comment (str): Comment for tensorboard writer

        model_idx (int): Index of training model

        use_cuda (bool): If true use cuda for computation
    """
    # Set device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # List of parameters for PopSAN parameters (save this for convenience)
    popsan_params = itertools.chain(ac.popsan.encoder.parameters(),
                                    ac.popsan.snn.parameters(),
                                    ac.popsan.decoder.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update)

    def _compute_policy_values(obs_pi, obs_q):
        # with torch.no_grad():
        actions_pred, log_pis = ac.popsan(obs_pi,batch_size=len(obs_pi))

        qs1 = ac.q1(obs_q, actions_pred)
        qs2 = ac.q2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** act_dim)
        return random_values - random_log_probs

    # Set up function for computing Spike-SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.popsan(o2, batch_size)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        # CQL addon
        random_actions = torch.FloatTensor(q1.shape[0] * 10, a.shape[-1]).uniform_(-1, 1).to(device)
        num_repeat = int(random_actions.shape[0] / o.shape[0])
        temp_states = o.unsqueeze(1).repeat(1, num_repeat, 1).view(o.shape[0] * num_repeat, o.shape[1])
        temp_next_states = o2.unsqueeze(1).repeat(1, num_repeat, 1).view(o2.shape[0] * num_repeat,
                                                                                  o2.shape[1])

        current_pi_values1, current_pi_values2 = _compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = _compute_policy_values(temp_next_states, temp_states)

        random_values1 = _compute_random_values(temp_states, random_actions, ac.q1).reshape(o.shape[0],
                                                                                                        num_repeat, 1)
        random_values2 = _compute_random_values(temp_states, random_actions, ac.q2).reshape(o.shape[0],
                                                                                                        num_repeat, 1)

        current_pi_values1 = current_pi_values1.reshape(o.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(o.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(o.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(o.shape[0], num_repeat, 1)

        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)

        assert cat_q1.shape == (o.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (o.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"

        cql1_scaled_loss = ((torch.logsumexp(cat_q1, dim=1).mean()) - q1.mean())
        cql2_scaled_loss = ((torch.logsumexp(cat_q2, dim=1).mean()) - q2.mean())


        loss_q = loss_q1 + loss_q2 + cql1_scaled_loss * 0.1+ cql2_scaled_loss * 0.1

        # Useful info for logging
        q_info = dict(Q1Vals=q1.to('cpu').detach().numpy(),
                      Q2Vals=q2.to('cpu').detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data, env_fn):
        o = data['obs']

        # env = env_fn()
        # mirror_observation, mirror_action = None, None
        # if hasattr(env, 'mirror_observation'):
        #     if env.clock_based:
        #         mirror_observation = env.mirror_clock_observation
        #     else:
        #         mirror_observation = env.mirror_observation
        #     
        # if hasattr(env, 'mirror_action'):
        #     mirror_action = env.mirror_action
        # 
        # if mirror_observation is not None and mirror_action is not None:
        #     # deterministic_actions = policy(obs_batch)
        #     deterministic_actions = ac.act(o, batch_size=batch_size, deterministic=True)
        #     if env.clock_based:
        #         mir_obs = mirror_observation(o.cpu(), env.clock_inds).to(device)
        #         mirror_actions = ac.act(mir_obs, batch_size=batch_size, deterministic=True)
        #     else:
        #         mirror_actions = ac.act(mirror_observation(o.cpu()).to(device),batch_size=batch_size,deterministic=True)
        #     mirror_actions = mirror_action(mirror_actions)
        #     mirror_loss = 0.4 * (deterministic_actions - mirror_actions).pow(2).mean()
        # else:
        #     mirror_loss = 0

        pi, logp_pi = ac.popsan(o, batch_size)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean() 

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.to('cpu').detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    popsan_mean_optimizer = Adam(popsan_params, lr=popsan_lr)
    pi_std_optimizer = Adam(ac.popsan.log_std_network.parameters(), lr=q_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, env_fn):

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        popsan_mean_optimizer.zero_grad()
        pi_std_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data, env_fn)
        loss_pi.backward()
        popsan_mean_optimizer.step()
        pi_std_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1,
                      deterministic)

    def test_agent():
        ###
        # compuate the return mean test reward
        ###
        test_reward_sum = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            if isinstance(o, tuple):
                o = o[0]
            # while not (d or (ep_len == max_ep_len)):
            while not d:
                # Take deterministic actions at test time
                a_test = get_action(replay_buffer.normalize_obs(o), True)
                if len(a_test.shape) > 1:
                    a_test = a_test.squeeze()
                res = test_env.step(a_test)
                if len(res) == 5:
                    o, r, d, _, _ = res
                else:
                    o, r, d, _ = res
                # o, r, d, _ = test_env.step(get_action(replay_buffer.normalize_obs(o), True))
                if isinstance(o, tuple):
                    o = o[0]
                o = o.squeeze()
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes

    ###
    # add tensorboard support and save rewards
    # Also create dir for saving parameters
    ###
    writer = SummaryWriter(comment="_" + tb_comment + "_" + str(model_idx))
    save_test_reward = []
    save_test_reward_steps = []
    try:
        os.mkdir("./sac_trained")
        print("Directory params Created")
    except FileExistsError:
        print("Directory params already exists")
    model_dir = "./sac_trained/spike-sac_" + tb_comment
    try:
        os.mkdir(model_dir)
        print("Directory ", model_dir, " Created")
    except FileExistsError:
        print("Directory ", model_dir, " already exists")

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    if isinstance(o, tuple):
        o = o[0]
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(replay_buffer.normalize_obs(o))
        else:
            a = np.random.rand(10) * 2 -1

        # Step the env
        # o2, r, d, _ = env.step(a)
        if len(a.shape) > 1:
            a = a.squeeze()
        if len(a.shape) == 0:
            a = a.reshape(-1)
        res = env.step(a)
        if len(res) == 5:
            o2, r, d, _, _ = res
        else:
            o2, r, d, _ = res
        if isinstance(o2, tuple):
            o2 = o2[0]
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            writer.add_scalar(tb_comment + '/Train-Reward', ep_ret, t + 1)
            o, ep_ret, ep_len = env.reset(), 0, 0
            if isinstance(o, tuple):
                o = o[0]

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch, env_fn=env_fn)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = t+1

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                ac.popsan.to('cpu')
                torch.save(ac.popsan.state_dict(),
                           model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')
                # print("Learned Mean for encoder population: ")
                # print(ac.popsan.encoder.mean.data)
                # print("Learned STD for encoder population: ")
                # print(ac.popsan.encoder.std.data)
                ac.popsan.to(device)
                print("Weights saved in ", model_dir + '/' + "model" + str(model_idx) + "_e" + str(epoch) + '.pt')

            # Test the performance of the deterministic version of the agent.
            test_mean_reward = test_agent()
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            writer.add_scalar(tb_comment + '/Test-Mean-Reward', test_mean_reward, t + 1)
            print("Model: ", model_idx, " Steps: ", t + 1, " Mean Reward: ", test_mean_reward)

    # Save Test Reward List
    pickle.dump([save_test_reward, save_test_reward_steps],
                open(model_dir + '/' + "model" + str(model_idx) + "_test_rewards.p", "wb+"))


def run():
    import math
    import argparse
    # env = gym.make('Pendulum-v1',max_episode_steps=300)
    traj = "walking"
    env_fn = env_factory('Cassie-v0',
                         simrate=50,
                         command_profile='clock',
                         input_profile='full',
                         learn_gains=False,
                         dynamics_randomization=True,
                         reward='iros_paper',
                         history=0,
                         mirror=True,
                         ik_baseline=False,
                         no_delta=True,
                         traj=traj)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='Pendulum-v1')
    # parser.add_argument('--encoder_pop_dim', type=int, default=10)
    # parser.add_argument('--decoder_pop_dim', type=int, default=10)
    # parser.add_argument('--encoder_var', type=float, default=0.15)
    # parser.add_argument('--start_model_idx', type=int, default=0)
    # parser.add_argument('--num_model', type=int, default=10)
    # parser.add_argument('--epochs', type=int, default=100)
    # args = parser.parse_args()
    epochs = 1000
    START_MODEL = 0
    NUM_MODEL = 10
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=10,
                     decoder_pop_dim=10,
                     mean_range=(-3, 3),
                     std=math.sqrt(0.15),
                     spike_ts=5,
                     device=torch.device('cuda'))
    COMMENT = "sac-popsan-cql-" + 'Cassie-v0' + "-encoder-dim-" + str(AC_KWARGS['encoder_pop_dim']) + \
              "-decoder-dim-" + str(AC_KWARGS['decoder_pop_dim'])
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = num * 10
        # spike_sac(lambda: gym.make(args.env), actor_critic=SpikeActorDeepCritic, ac_kwargs=AC_KWARGS,
        #           popsan_lr=1e-4, gamma=0.99, seed=seed, epochs=args.epochs,
        #           norm_clip_limit=3.0, tb_comment=COMMENT, model_idx=num, steps_per_epoch=1000)
        spike_sac(env_fn, actor_critic=SpikeActorDeepCritic, ac_kwargs=AC_KWARGS,
                  popsan_lr=1e-4, gamma=0.99, seed=seed, epochs=epochs,
                  norm_clip_limit=3.0, tb_comment=COMMENT, model_idx=num, steps_per_epoch=1000)
if __name__ == '__main__':
    run()