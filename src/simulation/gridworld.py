import random
from random import randrange
import numpy as np
import noise

from simulation.obstacles import Obstacle


class SimulationModel:
    """Model for the grid world"""

    def __init__(self, width, height, num_survivors, agents=None, world=None, autogen_config=None):
        """
        :param width: Width of the map
        :param height: Height of the map
        :param world: Default obstacles world in a grid
        """
        if agents is None:
            agents = []
        if world is None:
            world = [[Obstacle.Empty for _ in range(width)] for _ in range(height)]

        self._world = world
        self._width = width
        self._height = height

        self._agents = agents
        if autogen_config is not None:
            self.generate_terrain(autogen_config)

        self.generate_survivors(num_survivors)

        self.generate_agent_positions()

    def generate_noise_grid(self, scale, octaves, persistence, lacunarity):
        # Add a random offset to the perlin noise
        offset = random.randrange(0, 100000)
        noise_grid = ((noise.pnoise3(x / scale,
                                     y / scale,
                                     offset,
                                     octaves=octaves,
                                     persistence=persistence,
                                     lacunarity=lacunarity)
                       for x in range(self._width)) for y in range(self._height))
        return noise_grid

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
            self.start_fires(chosen_map_config["fire spread"])
        if "rocks" in chosen_map_config:
            self.noise_terrain(chosen_map_config["rocks"], Obstacle.Rocks)

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
                        and x not in (0, self.get_width()-1) \
                        and y not in (0, self.get_height()-1)\
                        and self.get_at_cell(x, y) == Obstacle.Empty:
                    self.set_at_cell(x, y, obs)

    def _get_all_blocks_type(self, block_types):
        """
        Find all blocks of a certain type
        :param block_types: List of blocks to search for
        :return: Set of (x, y) tuples
        """
        block_types = set(block_types)  # To speed up searches
        for x in range(self.get_width()):
            for y in range(self.get_width()):
                if self.get_at_cell(x, y) in block_types:
                    yield x, y

    def _choose_of_block_type(self, block_types, n):
        locs = list(self._get_all_blocks_type(block_types))
        locs_indices = np.arange(len(locs))
        choices = np.random.choice(locs_indices, n, replace=False)
        return [locs[choice] for choice in choices]

    def generate_survivors(self, num_survivors):
        for loc in self._choose_of_block_type([Obstacle.Empty], num_survivors):
            self.set_at_cell(loc[0], loc[1], Obstacle.Survivor)

    def generate_agent_positions(self):
        for i, loc in enumerate(self._choose_of_block_type([Obstacle.Empty], len(self._agents))):
            self._agents[i].set_pos(loc[0], loc[1])

    def start_fires(self, fire_spread_config):
        # Find all trees
        trees_to_burn = self._choose_of_block_type([Obstacle.Tree], fire_spread_config["starting points"])

        for loc in trees_to_burn:
            self.set_at_cell(loc[0], loc[1], Obstacle.TreeFire)

    def point_in_bounds(self, x, y):
        return 0 <= x < self._width and 0 <= y < self._height

    def get_at_cell(self, x, y):
        return self._world[y][x] if self.point_in_bounds(x, y) else Obstacle.OutsideMap

    def set_at_cell(self, x, y, obj):
        if not self.point_in_bounds(x, y):
            raise ValueError("(%d, %d) is out of bounds".format(x, y))
        self._world[y][x] = obj

    def get_area(self, left, right, top, bottom, agent_positions):
        """
        Gets a square area of the map.
        The grid is
        :param agent_positions:
        :param left: The left of the area (inclusive)
        :param right: Right of the area (inclusive)
        :param top: The top of the area (inclusive)
        :param bottom: Bottom of the area (inclusive)
        :return: np array of the area in the map
        """
        grid = np.array([[self.get_at_cell(x, y) if (x, y) not in agent_positions else Obstacle.Agent for x in
                          range(left, right + 1)] for y in range(top, bottom + 1)])
        return grid

    def agent_scan(self, agent, agent_positions):

        area = self.get_area(*agent.get_sight_area(), agent_positions)

        return np.rot90(area, k=-agent.get_rotation(), axes=(0, 1))

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
