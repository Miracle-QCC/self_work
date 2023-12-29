class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def remote_set_agent_idx(self, agent_idx):
        for remote in self.remotes:
            remote.send(('set_agent_idx', agent_idx))

    def remote_get_agent_idx(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_agent_idx', None))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

class RewardShapingEnv(VecEnvWrapper):
    """
    Wrapper for the Baselines vectorized environment, which
    modifies the reward obtained to be a combination of intrinsic
    (dense, shaped) and extrinsic (sparse, from environment) reward"""

    def __init__(self, env, reward_shaping_factor=0.0):
        super().__init__(env)
        self.reward_shaping_factor = reward_shaping_factor
        self.env_name = "Overcooked-v0"

        ### Set various attributes to false, than will then be overwritten by various methods

        # Whether we want to query the actual action method from the agent class,
        # or we use direct_action. Might change things if there is post-processing
        # of actions returned, as in the Human Model
        self.use_action_method = False

        # Fraction of self-play actions/trajectories (depending on value of self.trajectory_sp)
        self.self_play_randomization = 0.0

        # Whether SP randomization should be done on a trajectory level
        self.trajectory_sp = False

        # Whether the model is supposed to output the joint action for all agents (centralized policy)
        # Joint action models are currently deprecated.
        self.joint_action_model = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        # replace rew with shaped rew
        for env_num in range(self.num_envs):
            dense_reward = infos[env_num]['shaped_r']
            rew = list(rew)
            shaped_rew = rew[env_num] + float(dense_reward) * self.reward_shaping_factor
            rew[env_num] = shaped_rew

            if done[env_num]:
                # Log both sparse and dense rewards for episode
                sparse_ep_rew = infos[env_num]['episode']['ep_sparse_r']
                dense_ep_rew = infos[env_num]['episode']['ep_shaped_r']
                infos[env_num]['episode']['r'] = sparse_ep_rew + dense_ep_rew * self.reward_shaping_factor

        return obs, rew, done, infos

    def update_reward_shaping_param(self, reward_shaping_factor):
        """Takes in what fraction of the run we are at, and determines the reward shaping coefficient"""
        self.reward_shaping_factor = reward_shaping_factor