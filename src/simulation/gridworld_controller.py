from random import randrange, random

import noise
import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from simulation.agent import Agent
from simulation.gridworld_model import SimulationModel
from simulation.obstacles import Obstacle, is_flammable


class SimulationController:
    def __init__(self, width, height, num_survivors, num_agents,
                 sight, battery, reward_map, battery_costs, fire_spread, autogen_config):
        self._width = width
        self._height = height
        self._num_survivors = num_survivors
        self._num_agents = num_agents
        self._sight = sight
        self._battery = battery
        self._reward_map = reward_map
        self._battery_costs = battery_costs
        self._fire_spread = fire_spread
        self._autogen_config = autogen_config

        self.agents = None
        self.model = None

    def initialise(self):
        self.agents = [Agent(rot=0,
                             sight=self._sight,
                             battery=self._battery,
                             battery_costs=self._battery_costs)
                       for i in range(self._num_agents)]
        self.model = SimulationModel(self._width, self._height)

        if self._autogen_config is not None:
            self.generate_terrain(self._autogen_config)

        self.generate_survivors(self._num_survivors)
        self.generate_agent_positions()

        self.start_fires(self._fire_spread)

    def start_fires(self, fire_spread_config):
        # Find all trees
        trees_to_burn = self.model.choose_of_block_type([Obstacle.Tree], fire_spread_config["starting points"])

        self.model.add_burning_cells(trees_to_burn)

    def generate_agent_positions(self):
        for i, loc in enumerate(self.model.choose_of_block_type([Obstacle.HQ], len(self.agents))):
            self.agents[i].set_pos(loc[0], loc[1])

    @staticmethod
    def _pick_map(autogen_config):
        choices = list(autogen_config.values())
        probs = [i["chance"] for i in autogen_config.values()]
        return np.random.choice(choices, 1, p=probs)[0]

    def generate_terrain(self, autogen_config):
        # TODO make this weighted
        chosen_map_config = self._pick_map(autogen_config)

        if "trees" in chosen_map_config:
            self.noise_terrain(chosen_map_config["trees"], Obstacle.Tree)
        if "rocks" in chosen_map_config:
            self.noise_terrain(chosen_map_config["rocks"], Obstacle.Rocks)
        if "hq" in chosen_map_config:
            self.generate_hq(chosen_map_config["hq"])

    def generate_hq(self, hq_config):
        center_x = self.model.get_width() // 2
        center_y = self.model.get_height() // 2

        # Add clearing around HQ
        self.model.set_cells_box(center_x, center_y, hq_config["size"] + 4, hq_config["size"] + 4, Obstacle.Empty)

        # Add HQ ground for agents to spawn in
        self.model.set_cells_box(center_x, center_y, hq_config["size"], hq_config["size"], Obstacle.HQ)

    def generate_noise_grid(self, scale, octaves, persistence, lacunarity):
        # Add a random offset to the perlin noise
        offset = randrange(0, 100000)
        noise_grid = ((noise.pnoise3(x / scale,
                                     y / scale,
                                     offset,
                                     octaves=octaves,
                                     persistence=persistence,
                                     lacunarity=lacunarity)
                       for x in range(self._width)) for y in range(self._height))
        return noise_grid

    def noise_terrain(self, noise_config, obs):
        noise_grid = self.generate_noise_grid(noise_config["scale"],
                                              noise_config["octaves"],
                                              noise_config["persistence"],
                                              noise_config["lacunarity"])
        for x, row in enumerate(noise_grid):
            for y, cell in enumerate(row):
                # Leave 1 cell clear round the edge of the map
                # Only when the noise is above the threshold should there be trees
                if cell > noise_config["threshold"] \
                        and x not in (0, self.model.get_width() - 1) \
                        and y not in (0, self.model.get_height() - 1) \
                        and self.model.get_at_cell(x, y) == Obstacle.Empty:
                    self.model.set_at_cell(x, y, obs)

    def generate_survivors(self, num_survivors):
        for loc in self.model.choose_of_block_type([Obstacle.Empty], num_survivors):
            self.model.set_at_cell(loc[0], loc[1], Obstacle.Survivor)

    def get_observations(self, rew) -> MultiAgentDict:
        agent_positions = self.get_agent_positions()
        obs = {}
        for i, agent in enumerate(self.agents):
            terrain, agents = self.model.agent_scan(agent, agent_positions)
            obs[i] = {
                "terrain": terrain,
                "agents": agents
            }
            rew[i] += self.model.get_newly_explored() * self._reward_map["exploring"]
            assert terrain.shape == (self._sight * 2 + 1, self._sight * 2 + 1)
            assert agents.shape == (self._sight * 2 + 1, self._sight * 2 + 1)
        return obs

    def all_agents_dead(self):
        return [agent.is_dead() for i, agent in enumerate(self.agents)]

    def perform_actions(self, action_dict, rew):

        self.step_simulation()

        for i, agent in enumerate(self.agents):
            # Perform selected action
            if i in action_dict.keys() and not agent.is_dead():
                agent.actions()[action_dict[i]]()
                if self.model.get_at_cell(agent.get_x(), agent.get_y()) == Obstacle.Survivor:
                    rew[i] += self._reward_map["rescue"]
                    self.model.set_at_cell(agent.get_x(), agent.get_y(), Obstacle.Empty)
                if self.model.get_at_cell(agent.get_x(), agent.get_y()) == Obstacle.Tree:
                    rew[i] += self._reward_map["hit tree"]
                    agent.kill()
                # if self.model.get_at_cell(agent.get_x(), agent.get_y()) == Obstacle.OutsideMap:
                # rew[i] -=
                # If it goes outside map, punish it
        return rew

    def step_simulation(self):
        self.spread_fire()

    def spread_fire(self):
        total_samples = [[0 for _ in range(self.model.get_width())] for _ in range(self.model.get_height())]
        # Choose some of the currently burning trees, to spread
        for pos in self.model.get_burning_cells():
            if random() < self._fire_spread["rate"]:
                new_samples = np.random.multivariate_normal(
                    pos,
                    self._fire_spread["covariance"],
                    1)
                for sample in new_samples:
                    x = round(sample[0])
                    y = round(sample[1])
                    if self.model.in_bounds(x, y):
                        total_samples[y][x] += 1
        for y, col in enumerate(total_samples):
            for x, sample in enumerate(col):
                if is_flammable(sample >= 1 and self.model.get_at_cell(x, y)):
                    self.model.add_burning_cells([(x, y)])

    def get_agent_positions(self):
        positions = {}
        for agent in self.agents:
            positions[(agent.get_x(), agent.get_y())] = agent
        return positions

    def get_sight_range(self):
        return self._sight
