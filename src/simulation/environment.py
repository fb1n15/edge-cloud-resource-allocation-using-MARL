from typing import Tuple

from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from simulation.agent import Agent
from simulation.gridworld_controller import SimulationController
from simulation.obstacles import Obstacle


class GridWorldEnv(MultiAgentEnv):
    """Logic for managing the simulation"""
    VERSION = 2  # Increment each time there are non-backwards compatible changes made to simulation

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
        self.observation_space = Box(low=0, high=len(Obstacle), shape=((config["sight"] * 2 + 1) ** 2,))

    def reset(self) -> MultiAgentDict:
        self.controller.initialise()
        return self.controller.get_observations()

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        rew = self.controller.perform_actions(action_dict)
        obs = self.controller.get_observations()
        done = {"__all__": all(self.controller.all_agents_dead())}

        return obs, rew, done, {}
