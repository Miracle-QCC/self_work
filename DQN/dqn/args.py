class Args:
    def __init__(self):
        self.with_per = True  # 是否使用优先经验回放（Prioritized Experience Replay, PER）
        self.consecutive_frames = 1# 用于构建状态的连续帧数
        self.dueling = False  # 是否使用Dueling DQN架构
        self.nb_episodes = 100 # 要执行的训练回合（episode）数量
        self.batch_size = 16 # 用于训练智能体（agent）的经验回放批次的大小
        self.render = False  # 在训练过程中是否渲染环境
        self.gather_stats = True  # 是否在训练过程中收集统计信息
        self.type="ddqn"
        self.env="sumo"
# 创建 Args 对象并设置参数
args = Args()
args.with_per = True
args.consecutive_frames = 4
args.dueling = False
args.nb_episodes = 100
args.batch_size = 32
args.render = False
args.gather_stats = True
