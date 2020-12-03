from enum import Enum


class Obstacle(Enum):
    """Obstacle"""
    TestObstacle = -1
    OutsideMap = 0
    Empty = 1
    Tree = 2
    Survivor = 3
    Agent = 4
