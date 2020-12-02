from typing import Tuple

from gym.spaces import Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from simulation.agent import Agent
from simulation.gridworld import GridWorldModel
from simulation.obstacles import Obstacle
from simulation.survivor import Survivor


class SimulationController:
    def __init__(self, width, height, num_survivors, num_agents, sight, battery):
        self._width = width
        self._height = height
        self._num_survivors = num_survivors
        self._num_agents = num_agents
        self._sight = sight
        self._battery = battery

        self.agents = None
        self.model = None

    def initialise(self):
        self.agents = [Agent(x=i + int(self._width / 2), y=i + int(self._height / 2), rot=0, sight=self._sight,
                             battery=self._battery) for i in
                       range(self._num_agents)]
        self.model = GridWorldModel(self._width, self._height, self._num_survivors)

    def get_observations(self) -> MultiAgentDict:
        agent_positions = self.get_agent_positions()
        obs = {}
        for i, agent in enumerate(self.agents):
            agent_obs = self.model.agent_scan(agent).flatten()
            obs[i] = [x.value if isinstance(x, Obstacle) else len(Obstacle)
                      for x in agent_obs]
            assert len(agent_obs) == (self._sight * 2 + 1) ** 2
        return obs

    def all_agents_dead(self):
        return [agent.is_dead() for i, agent in enumerate(self.agents)]

    def perform_actions(self, action_dict):
        # Set all rewards at 0 to start with
        rew = {i: 0 for i in range(len(self.agents))}
        for i, agent in enumerate(self.agents):
            # Perform selected action
            if i in action_dict.keys():
                agent.actions()[action_dict[i]]()
                if isinstance(self.model.get_at_cell(agent.get_x(), agent.get_y()), Survivor):
                    rew[i] += 1
                    self.model.set_at_cell(agent.get_x(), agent.get_y(), Obstacle.Empty)
                if self.model.get_at_cell(agent.get_x(), agent.get_y()) == Obstacle.OutsideMap:
                    rew[i] -= 10
                    # If it goes outside map, punish it
        return rew

    def get_agent_positions(self):
        positions = {}
        for agent in self.agents:
            positions[(agent.get_x(), agent.get_y())] = agent
        return positions


class GridWorldEnv(MultiAgentEnv):
    """Logic for managing the simulation"""

    def __init__(self, config):

        self.controller = SimulationController(config["width"],
                                               config["height"],
                                               config["num_survivors"],
                                               config["num_agents"],
                                               config["sight"],
                                               config["battery"])

        self.action_space = Discrete(len(Agent().actions()))
        self.observation_space = Box(low=0, high=len(Obstacle) + 1, shape=((config["sight"] * 2 + 1) ** 2,))

    def reset(self) -> MultiAgentDict:
        self.controller.initialise()
        return self.controller.get_observations()

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        rew = self.controller.perform_actions(action_dict)
        obs = self.controller.get_observations()
        done = {"__all__": all(self.controller.all_agents_dead())}

        return obs, rew, done, {}
