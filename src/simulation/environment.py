from pprint import pprint
from typing import Tuple

from gym.spaces import Discrete, Box, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from simulation.entities import Agent
from simulation.gridworld_controller import SimulationController
from simulation.observables import Obstacle


class GridWorldEnv(MultiAgentEnv):
    """Logic for managing the simulation"""
    VERSION = 3  # Increment each time there are non-backwards compatible changes made to simulation

    def __init__(self, config):
        self.controller = SimulationController(config["width"],
                                               config["height"],
                                               config["num_survivors"],
                                               config["num_agents"],
                                               config["sight"],
                                               config["battery"],
                                               config["rewards"],
                                               config["battery costs"],
                                               config["fire spread"],
                                               config["autogen config"])

        self.action_space = Discrete(len(Agent().actions()))
        self.observation_space = Dict({
            "terrain": Box(low=0, high=len(Obstacle), shape=((config["sight"] * 2 + 1), (config["sight"] * 2 + 1))),
            "agents": Box(low=0, high=1, shape=((config["sight"] * 2 + 1), (config["sight"] * 2 + 1))),
            "survivors": Box(low=0, high=1, shape=((config["sight"] * 2 + 1), (config["sight"] * 2 + 1))),
        })

    def _empty_reward_map(self):
        return {i: 0 for i in range(len(self.controller.agents))}

    def reset(self) -> MultiAgentDict:
        self.controller.initialise()
        # Discard observation based rewards for first iteration
        return self.controller.get_observations(self._empty_reward_map())

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # Set all rewards at 0 to start with
        rew = self._empty_reward_map()
        self.controller.perform_actions(action_dict, rew)
        obs = self.controller.get_observations(rew)
        done = {"__all__": self.controller.all_agents_dead()}

        return obs, rew, done, {}

    def num_agents_crashed(self):
        return self.controller.num_agents_crashed()

    def get_survivors_rescued(self):
        return self.controller.get_survivors_rescued()
