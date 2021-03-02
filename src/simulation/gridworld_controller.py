from random import randrange, random

import noise
import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from simulation.entities import Agent, Survivor
from simulation.gridworld_model import SimulationModel
from simulation.observables import Obstacle, is_flammable, is_collidable


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
        self.survivors = []

        self.survivors_rescued = 0
        self.agents_crashed = 0

    def initialise(self):
        self.agents = [Agent(rot=0,
                             sight=self._sight,
                             battery=self._battery,
                             battery_costs=self._battery_costs)
                       for i in range(self._num_agents)]
        self.survivors = []
        self.model = SimulationModel(self._width, self._height)

        if self._autogen_config is not None:
            self.generate_terrain(self._autogen_config)

        self.generate_survivors(self._num_survivors)
        self.generate_agent_positions()

        self.start_fires(self._fire_spread)

        self.survivors_rescued = 0
        self.agents_crashed = 0

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
            self.survivors.append(Survivor(loc[0], loc[1]))

    def get_area(self, left, right, top, bottom, agent_positions, survivor_positions):
        """
        Gets a square area of the map.
        The grid is
        :param survivor_positions:
        :param agent_positions:
        :param left: The left of the area (inclusive)
        :param right: Right of the area (inclusive)
        :param top: The top of the area (inclusive)
        :param bottom: Bottom of the area (inclusive)
        :return: np array of the area in the map
        """
        terrain = np.array([[
            self.model.get_at_cell(x, y).value
            for x in range(left, right + 1)]
            for y in range(top, bottom + 1)]
        )

        agents = np.array([[
            1 if (x, y) in agent_positions else 0
            for x in range(left, right + 1)]
            for y in range(top, bottom + 1)]
        )

        survivors = np.array([[
            1 if (x, y) in survivor_positions else 0
            for x in range(left, right + 1)]
            for y in range(top, bottom + 1)]
        )
        return terrain, agents, survivors

    @staticmethod
    def _rotate_view(view, rot):
        return np.rot90(view, k=rot, axes=(0, 1))

    def agent_scan(self, agent, agent_positions, survivor_positions):
        agent_sight = agent.get_sight_area()
        terrain, agents, survivors = self.get_area(*agent_sight, agent_positions, survivor_positions)
        self.model.explore_cells(*agent_sight)
        return \
            self._rotate_view(terrain, -agent.get_rotation()),\
            self._rotate_view(agents, -agent.get_rotation()),\
            self._rotate_view(survivors, -agent.get_rotation()),

    def get_observations(self, rew) -> MultiAgentDict:
        agent_positions = self.get_agent_positions()
        survivor_positions = self.get_survivor_positions()
        obs = {}
        for i, agent in enumerate(self.agents):
            terrain, agents, survivors = self.agent_scan(agent, agent_positions, survivor_positions)
            obs[i] = {
                "terrain": terrain,
                "agents": agents,
                "survivors": survivors
            }
            rew[i] += self.model.get_newly_explored() * self._reward_map["exploring"]
            assert terrain.shape == (self._sight * 2 + 1, self._sight * 2 + 1)
            assert agents.shape == (self._sight * 2 + 1, self._sight * 2 + 1)
            assert survivors.shape == (self._sight * 2 + 1, self._sight * 2 + 1)
        return obs

    def num_agents_dead(self):
        return [agent.is_dead() for i, agent in enumerate(self.agents)].count(True)

    def all_agents_dead(self):
        return self.num_agents_dead() == len(self.agents)

    def num_agents_crashed(self):
        return self.agents_crashed

    def perform_actions(self, action_dict, rew):

        self.step_simulation()

        for i, agent in enumerate(self.agents):
            # Perform selected action
            if i in action_dict.keys() and not agent.is_dead():
                agent.actions()[action_dict[i]]()
                if (agent.get_x(), agent.get_y()) in self.get_survivor_positions():
                    rew[i] += self._reward_map["rescue"]
                    self.rescue_survivor(agent.get_x(), agent.get_y())
                if is_collidable(self.model.get_at_cell(agent.get_x(), agent.get_y())):
                    rew[i] += self._reward_map["hit tree"]
                    self.kill_agent(agent)
                # if self.model.get_at_cell(agent.get_x(), agent.get_y()) == Obstacle.OutsideMap:
                # rew[i] -=
                # If it goes outside map, punish it
        return rew

    def kill_agent(self, agent):
        self.agents_crashed += 1
        agent.kill()

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

    def get_survivor_positions(self):
        positions = {}
        for survivor in self.survivors:
            positions[(survivor.get_x(), survivor.get_y())] = survivor
        return positions

    def remove_survivor_at(self, x, y):
        """Filter out any survivors on this cell"""
        self.survivors = [survivor for survivor in self.survivors
                          if survivor.get_x() != x and survivor.get_y() != y]

    def get_survivors_rescued(self):
        return self.survivors_rescued

    def rescue_survivor(self, x, y):
        self.remove_survivor_at(x, y)
        self.survivors_rescued += 1

    def get_sight_range(self):
        return self._sight
