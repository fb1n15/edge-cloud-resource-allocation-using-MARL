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
        x_rand = random.randrange(0, 1000)
        y_rand = random.randrange(0, 1000)
        noise_grid = ((noise.pnoise2(x / scale + x_rand,
                                     y / scale + y_rand,
                                     octaves=octaves,
                                     persistence=persistence,
                                     lacunarity=lacunarity)
                       for x in range(self._width)) for y in range(self._height))
        return noise_grid

    def generate_terrain(self, autogen_config):
        chosen_map, chosen_map_config = random.choice(list(autogen_config.items()))

        if "trees" in chosen_map_config:
            self.generate_trees(chosen_map_config["trees"])

    def generate_trees(self, tree_config):
        noise_grid = self.generate_noise_grid(tree_config["scale"],
                                              tree_config["octaves"],
                                              tree_config["persistence"],
                                              tree_config["lacunarity"])
        for x, row in enumerate(noise_grid):
            for y, cell in enumerate(row):
                # Leave 1 cell clear round the edge of the map
                # Only when the noise is above the threshold should there be trees
                if cell > tree_config["threshold"] \
                        and x not in (0, self.get_width()-1) \
                        and y not in (0, self.get_height()-1):
                    self.set_at_cell(x, y, Obstacle.Tree)

    def generate_survivors(self, num_survivors):
        i = 0
        while i < num_survivors:
            x = randrange(self._width)
            y = randrange(self._height)
            if self.get_at_cell(x, y) == Obstacle.Empty:
                self.set_at_cell(x, y, Obstacle.Survivor)
                i += 1

    def generate_agent_positions(self):
        for agent in self._agents:
            for i in range(100000):  # Try until it cannot find any where - to prevent infinite loop
                x = randrange(self._width)
                y = randrange(self._height)
                if self.get_at_cell(x, y) == Obstacle.Empty:
                    agent.set_pos(x, y)

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
