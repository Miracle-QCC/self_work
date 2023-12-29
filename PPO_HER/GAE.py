def compute_GAE(self, mb_rewards, mb_values, mb_dones):
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self.model.value(self.obs, S=self.states, M=self.dones)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(self.nsteps)):  # 倒序实现，便于进行递归
        if t == self.nsteps - 1:  # 如果是最后一步，要判断当前是否是终止状态，如果是，next_value就是0
            nextnonterminal = 1.0 - self.dones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_dones[t + 1]
            nextvalues = mb_values[t + 1]
        delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
        mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam