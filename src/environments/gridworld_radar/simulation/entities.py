from abc import ABC, abstractmethod


class Entity(ABC):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def set_pos(self, x, y):
        self._x = x
        self._y = y


class Agent(Entity):
    def __init__(self, _id, controller=None, x=0, y=0, rot=0, sight=0, battery=100, battery_costs=None):
        """
        :param x: x position relative to top left
        :param y: y position relative to top left
        :param rot: rads clockwise from North
        :param sight: how far the agent can see
        """
        super(Agent, self).__init__(x, y)

        self.id = _id

        if battery_costs is None:
            battery_costs = {}
        self._rot = rot
        self._sight = sight
        self._battery = battery
        self._battery_costs = battery_costs
        self._dead = False
        self._controller = controller

    def rotate_left(self):
        """
        Rotate the agent left
        """
        self._rot = (self._rot - 1) % 4
        self.drain_battery("rotate left")

    def rotate_right(self):
        """
        Rotate the agent right
        """
        self._rot = (self._rot + 1) % 4
        self.drain_battery("rotate right")

    def advance(self):
        if self._rot == 0:
            self._y -= 1
        elif self._rot == 1:
            self._x += 1
        elif self._rot == 2:
            self._y += 1
        elif self._rot == 3:
            self._x -= 1
        self.drain_battery("advance")

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
        1 -> Right
        2 -> Down
        3 -> Left
        """
        return self._rot

    def get_sight_dist(self):
        return self._sight

    def is_dead(self):
        return self._battery <= 0 or self._dead

    def kill(self):
        self._dead = True

    def drain_battery(self, cost_type):
        self._battery -= self._battery_costs[cost_type]

    @abstractmethod
    def actions(self):
        """The actions this agent can take"""


class RadarDrone(Agent):

    def mark(self):
        self._controller.model.mark_cell(self._x, self._y)

    def actions(self):
        return [self.rotate_left, self.rotate_right, self.advance, self.mark]


class RescueDrone(Agent):

    def actions(self):
        return [self.rotate_left, self.rotate_right, self.advance]


class Survivor(Entity):
    def __init__(self, x, y, alive=True):
        super(Survivor, self).__init__(x, y)
        self.alive = alive

    def is_dead(self):
        return not self.alive
