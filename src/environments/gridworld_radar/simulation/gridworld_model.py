import numpy as np

from environments.gridworld_radar.simulation.observables import Obstacle


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
        self._entities = []

        self._width = width
        self._height = height

        self._burning_cells = {}

        self._explored_cells = [[False for _ in range(width)] for _ in range(height)]
        self._newly_explored = 0

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

    def explore_cells(self, left, right, top, bottom):
        for x in range(left, right+1):
            for y in range(top, bottom+1):
                if self.in_bounds(x, y) and not self._explored_cells[y][x]:
                    # If unexplored
                    self._explored_cells[y][x] = True
                    self._newly_explored += 1

    def percentage_explored(self):
        return sum(row.count(True) for row in self._explored_cells)/(self.get_width() * self.get_height())

    def get_newly_explored(self):
        newly_explored = self._newly_explored
        # Reset to 0
        self._newly_explored = 0
        return newly_explored

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height
