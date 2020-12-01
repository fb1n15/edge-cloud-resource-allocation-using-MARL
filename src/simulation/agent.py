import numpy as np


class Agent:

    def __init__(self, x=0, y=0, rot=0, sight=0, battery=100):
        """
        :param x: x position relative to top left
        :param y: y position relative to top left
        :param rot: rads clockwise from North
        :param sight: how far the agent can see
        """
        self._x = x
        self._y = y
        self._rot = rot
        self._sight = sight
        self._battery = battery

    def rotate_left(self):
        """
        Rotate the agent left
        """
        self._rot = (self._rot - 1) % 4
        self.drain_battery(1)

    def rotate_right(self):
        """
        Rotate the agent right
        """
        self._rot = (self._rot + 1) % 4
        self.drain_battery(1)

    def advance(self):
        if self._rot == 0:
            self._y -= 1
        elif self._rot == 1:
            self._x += 1
        elif self._rot == 2:
            self._y += 1
        elif self._rot == 3:
            self._x -= 1
        self.drain_battery(2)

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

    def get_sight_dist(self):
        return self._sight

    def actions(self):
        return [self.rotate_left, self.rotate_right, self.advance]

    def is_dead(self):
        return self._battery <= 0

    def drain_battery(self, amount):
        self._battery -= amount

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y
