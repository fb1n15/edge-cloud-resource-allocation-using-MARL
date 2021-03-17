from typing import Tuple

from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from environments.gridworld_radar.simulation.entities import Agent
from environments.gridworld_radar.simulation.gridworld_controller import SimulationController


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

    @staticmethod
    def get_action_space():
        return Discrete(len(Agent("dummy").actions()))

    @staticmethod
    def get_observation_space(config):
        return Box(low=0, high=1, shape=((config["sight"] * 2 + 1), (config["sight"] * 2 + 1), 3))

    def _empty_reward_map(self):
        return {agent.id: 0 for agent in self.controller.agents}

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

    def get_map_explored(self):
        return self.controller.get_map_explored()
