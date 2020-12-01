from random import randrange
import numpy as np

from simulation.survivor import Survivor
from simulation.obstacles import Obstacle


class GridWorldModel:
    """Model for the grid world"""

    def __init__(self, width, height, num_survivors, agents=None, world=None):
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

        self.generate_survivors(num_survivors)

    def generate_survivors(self, num_survivors):
        i = 0
        while i < num_survivors:
            x = randrange(self._width)
            y = randrange(self._height)
            if self.get_at_cell(x, y) == Obstacle.Empty:
                self.set_at_cell(x, y, Survivor())
                i += 1

    def point_in_bounds(self, x, y):
        return 0 <= x < self._width and 0 <= y < self._height

    def get_at_cell(self, x, y):
        return self._world[y][x] if self.point_in_bounds(x, y) else Obstacle.OutsideMap

    def set_at_cell(self, x, y, obj):
        if not self.point_in_bounds(x, y):
            raise ValueError("(%d, %d) is out of bounds".format(x, y))
        self._world[y][x] = obj

    def get_area(self, left, right, top, bottom):
        """
        Gets a square area of the map.
        The grid is
        :param left: The left of the area (inclusive)
        :param right: Right of the area (inclusive)
        :param top: The top of the area (inclusive)
        :param bottom: Bottom of the area (inclusive)
        :return: np array of the area in the map
        """
        grid = np.array([[self.get_at_cell(x, y) for x in range(left, right+1)] for y in range(top, bottom+1)])
        return grid

    def agent_scan(self, agent):

        area = self.get_area(*agent.get_sight_area())

        return np.rot90(area, k=-agent.get_rotation(), axes=(0, 1))

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
