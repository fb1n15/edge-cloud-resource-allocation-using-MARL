import numpy as np


class Agent:

    def __init__(self, x, y, rot, sight):
        """
        :param x: x position relative to top left
        :param y: y position relative to top left
        :param rot: rads clockwise from North
        :param model: the gridworld model
        """
        self._x = x
        self._y = y
        self._rot = rot
        self._sight = sight

    def rotate_left(self):
        """
        Rotate the agent left
        """
        self._rot = (self._rot - 1) % 4

    def rotate_right(self):
        """
        Rotate the agent right
        """
        self._rot = (self._rot + 1) % 4

    def advance(self):
        if self._rot == 0:
            self._y -= 1
        elif self._rot == 1:
            self._x += 1
        elif self._rot == 2:
            self._y += 1
        elif self._rot == 3:
            self._x -= 1

    def get_sight_area(self):
        """
        Find the box for the sight area
        :return:
        """

        return (self._x - self._sight,
                self._x + self._sight,
                self._y - self._sight,
                self._y + self._sight)

    def get_rotation(self):
        """
        0 -> Up
        1 -> Left
        2 -> Down
        3 -> Right
        """
        return self._rot
