import traci
import numpy as np
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
class Environment:
    def __init__(self,sumo_cfg_file):
        self.sumo_cfg_file=sumo_cfg_file
        self.sumo_binary='sumo-gui'
        self.min_safe_distance=50#最小安全距离
        self.max_step=40 #sumo  semo2 40
        self.targetID="vehicle3"

    def reset(self):
        traci.start([r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe", "-c",
                     r"D:\PycharmProjects\pythonProject2\DDQN\sumo\example.sumocfg"])
        while self.targetID not in traci.vehicle.getIDList():
            traci.simulationStep()
        state = self.get_state()
        self.lane_index=traci.vehicle.getLaneIndex(self.targetID)
        self.steps = 0
        self.done = False
        return state
    def reset2(self):
        traci.start([r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe", "-c",
                     r"D:\PycharmProjects\pythonProject2\DDQN\sumo2\example1.sumocfg"])
        while self.targetID not in traci.vehicle.getIDList():
            traci.simulationStep()
        state = self.get_state()
        self.lane_index = traci.vehicle.getLaneIndex(self.targetID)
        self.steps = 0
        self.done = False
        return state

    def get_state(self):
        if self.targetID not in traci.vehicle.getIDList():
            return [-1.0, -1.0, -1.0]
        if self.targetID in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(self.targetID)
        else:
            print(f"Vehicle '{self.targetID}' does not exist in the simulation.")
            speed = 0.0
        lane_position = traci.vehicle.getLanePosition(self.targetID)
        leader_info = traci.vehicle.getLeader(self.targetID, 500)
        distance = leader_info[1] if leader_info is not None else 500.0
        state = [speed, lane_position, distance]
        return state
    def step(self,action,state):
        if self.targetID not in traci.vehicle.getIDList():
            print(f"Vehicle '{self.targetID}' is not in simulation during action execution.")
            return np.array(state), 0, True,{}
        else:
            current_lane_index = traci.vehicle.getLaneIndex(self.targetID)
            max_lane_index = 1
            if action == 1:  # 右侧道
                traci.vehicle.changeLane("vehicle3", min(current_lane_index+1, max_lane_index), 5)
            elif action == 2:  # 左车道
                traci.vehicle.changeLane("vehicle3", max(current_lane_index-1, max_lane_index), 5)
            traci.simulationStep()
            next_state = self.get_state()
            reward = self.get_reward(next_state)
            done = self.get_done(next_state)

            self.steps += 1
            return next_state, reward, done,{}


    def get_reward(self,state):
        # 奖励2：适时变道
        old_lane_index = self.lane_index
        vehicle_ids = traci.vehicle.getIDList()
        speed_reward = 0.0
        if self.targetID in vehicle_ids:
            # 奖励1：保持高速度
            speed = traci.vehicle.getSpeed(self.targetID)
            self.lane_index = traci.vehicle.getLaneIndex(self.targetID)
            speed_reward = speed / traci.vehicle.getMaxSpeed(self.targetID) * 0.1
        else:
            print(f"Vehicle {self.targetID} not found in the simulation.")
        lane_change_reward = 0.0
        if old_lane_index != self.lane_index:  # 表示车辆变道了
            # 使用traci.vehicle.getLeader获得前方车辆的信息
            leader_info = traci.vehicle.getLeader(self.targetID, self.min_safe_distance)
            if leader_info is not None:  # 如果有前车
                distance_to_lead = leader_info[1]
                if distance_to_lead < self.min_safe_distance:  # 如果前车距离过近，奖励变道
                    lane_change_reward = 2
                else:  # 否则，惩罚不必要的变道
                    lane_change_reward = -2
        # 奖励3：避免碰撞
        collision_reward = 0.0
        if traci.simulation.getCollidingVehiclesNumber() > 0:  # 如果发生了碰撞
            collision_reward = -100.0  # 大幅惩罚

        # position_list = list(traci.vehicle.getIDList())  # 获取所有车辆ID列表，并按照位置排序（假设路网是水平方向）
        # position_list.sort(key=lambda x: traci.vehicle.getPosition(x)[0], reverse=True)
        # if self.targetID == position_list[0]:  # 如果目标车辆是第一辆车，说明达到了目标
        #     p_reward = 30.0  # 目标奖励，给予很大的正奖励
        # # 组合奖励
        reward = speed_reward + lane_change_reward + collision_reward#+p_reward
        return reward

    def close(self):
        traci.close()

    def get_done(self,state):
        if self.steps >= self.max_step:  # 超过最大步数，终止
           self.done = True
        if state==[-1,-1,-1]:
            self.done=True

        return self.done
