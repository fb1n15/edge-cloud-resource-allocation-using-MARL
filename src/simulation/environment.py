from typing import Tuple

from gym.spaces import Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from simulation.agent import Agent
from simulation.gridworld import GridWorldModel
from simulation.obstacles import Obstacle
from simulation.survivor import Survivor


class GridWorldEnv(MultiAgentEnv):
    """Logic for managing the simulation"""

    def __init__(self, config):
        self._width = config["width"]
        self._height = config["height"]
        self._num_survivors = config["num_survivors"]
        self._num_agents = config["num_agents"]
        self._sight = config["sight"]
        self._battery = config["battery"]
        # self._start_world = config.get("start_world", [[Obstacle.Empty for _ in range(self._width)] for _ in range(self._height)])

        self.action_space = Discrete(len(Agent().actions()))
        self.observation_space = Box(low=0, high=len(Obstacle) + 1, shape=((self._sight * 2 + 1) ** 2,))

    def reset(self) -> MultiAgentDict:
        self._agents = [Agent(x=i+int(self._width/2), y=i+int(self._height/2), rot=0, sight=self._sight, battery=self._battery) for i in
                        range(self._num_agents)]
        self._model = GridWorldModel(self._width, self._height, self._num_survivors)
        return self._get_observations()

    def _get_observations(self) -> MultiAgentDict:
        obs = {}
        for i, agent in enumerate(self._agents):
            agent_obs = self._model.agent_scan(agent).flatten()
            obs[i] = [x.value if isinstance(x, Obstacle) else len(Obstacle)
                      for x in agent_obs]
            assert len(agent_obs) == (self._sight * 2 + 1) ** 2
        return obs

    def _all_agents_dead(self):
        return [agent.is_dead() for i, agent in enumerate(self._agents)]

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # Set all rewards at 0 to start with
        rew = {i: 0 for i in range(len(self._agents))}

        for i, agent in enumerate(self._agents):
            # Perform selected action
            if i in action_dict.keys():
                agent.actions()[action_dict[i]]()
                if isinstance(self._model.get_at_cell(agent.get_x(), agent.get_y()), Survivor):
                    rew[i] += 1
                    self._model.set_at_cell(agent.get_x(), agent.get_y(), Obstacle.Empty)
        obs = self._get_observations()
        done = {}
        done["__all__"] = all(self._all_agents_dead())

        return obs, rew, done, {}
