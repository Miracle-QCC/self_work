import random

import highway_env
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


LANES = 2
ANGLE = 0
START = 0
LENGHT = 200
SPEED_LIMIT = 30
SPEED_REWARD_RANGE = [10, 30]
COL_REWARD = -1
HIGH_SPEED_REWARD = 0
RIGHT_LANE_REWARD = 0
DURATION = 100.0


class myEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "vehicles_count": 2,
                        "features": ["x", "y", "vx", "vy"],
                        "absolute": True,
                    },
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                    },
                },

                "reward_speed_range": SPEED_REWARD_RANGE,
                "simulation_frequency": 20,
                "policy_frequency": 20,
                "centering_position": [0.3, 0.5],
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()


    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(LANES, speed_limit=SPEED_LIMIT),
            np_random=self.np_random,
            record_history=False,
        )

    def _create_vehicles(self) -> None:

        self.controlled_vehicles = []
        vehicle = Vehicle.create_random(self.road, speed=23, lane_id=1, spacing=0.3)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
        )
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        vehicle = Vehicle.create_random(self.road, speed=30, lane_id=1, spacing=0.35)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
        )
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        reward = 0

        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )

        if self.vehicle.crashed:
            reward = -1
        elif lane == 0:
            reward += 1

        reward = 0 if not self.vehicle.on_road else reward

        return reward

    def _is_terminated(self) -> bool:
        return (
                self.vehicle.crashed
                or self.time >= DURATION
                or (False and not self.vehicle.on_road)
        )

    def step(self, action: Action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, info

if __name__ == '__main__':
    env = myEnv(render_mode="human")
    obs = env.reset()

    eposides = 10
    rewards = 0
    for eq in range(eposides):
        obs = env.reset()
        env.render()
        done = False
        while not done:
            # action = env.action_space.sample()
            action1 = random.sample([0,1,2,3,4], 1)[0]
            action2 = random.sample([0,1,2,3,4], 1)[0]
            action = (action1,action2)

            obs, reward, done, info = env.step(action)
            env.render()
            rewards += reward
        print(rewards)
