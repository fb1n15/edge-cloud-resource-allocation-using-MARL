import random
from random import randrange
import numpy as np
import noise

from simulation.obstacles import Obstacle


class SimulationModel:
    """Model for the grid world"""

    def __init__(self, width, height, world=None):
        """
        :param width: Width of the map
        :param height: Height of the map
        :param world: Default obstacles world in a grid
        """
        if world is None:
            world = [[Obstacle.Empty for _ in range(width)] for _ in range(height)]

        self._world = world

        self._width = width
        self._height = height

        self._burning_cells = {}

    def get_burning_cells(self):
        return set(self._burning_cells.keys())

    def remove_burning_cells(self, pos):
        self._burning_cells.pop(pos)

    def add_burning_cells(self, to_burn):
        """Add things to fast find list, and set cell types"""
        for loc in to_burn:
            if not self.is_cell_burning(loc[0], loc[1]):
                self._burning_cells[loc] = 0

    def is_cell_burning(self, x, y):
        return (x, y) in self._burning_cells

    def in_bounds(self, x, y):
        return 0 <= x < self.get_width() and 0 <= y < self.get_height()

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

    def choose_of_block_type(self, block_types, n):
        """Choose n blocks of a type from the map, and return their coordinates"""
        locs = list(self._get_all_blocks_type(block_types))
        locs_indices = np.arange(len(locs))
        choices = np.random.choice(locs_indices, n, replace=False)
        return [locs[choice] for choice in choices]

    def point_in_bounds(self, x, y):
        return 0 <= x < self._width and 0 <= y < self._height

    def get_at_cell(self, x, y):
        return self._world[y][x] if self.point_in_bounds(x, y) else Obstacle.OutsideMap

    def set_at_cell(self, x, y, obj):
        if not self.point_in_bounds(x, y):
            raise ValueError("(%d, %d) is out of bounds".format(x, y))
        self._world[y][x] = obj

    def set_cells_box(self, center_x, center_y, width, height, obstacle):
        """Sets a block of size width, height to a certain obstacle type"""

        for x_offset in range(-width//2, width//2+1):
            for y_offset in range(-height//2, height//2+1):
                x = center_x + x_offset
                y = center_y + y_offset
                self.set_at_cell(x, y, obstacle)

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
