from abc import ABC


class Obstacle(ABC):
    """Obstacle"""


class Tree(Obstacle):
    """Tree obstacle"""


class OutsideMap(Obstacle):
    """OutsideMap"""
