from enum import Enum


class Obstacle(Enum):
    """Obstacles
    Trees can be cleared by the drones, but costs a lot of energy
    Rocks cannot be passed by ground drones, and will destroy the drone
    Road
    """
    TestObstacle = -1
    Survivor = 0
    Agent = 1
    OutsideMap = 2
    Empty = 3
    Tree = 4
    BurnedTree = 6
    Rocks = 7
    Road = 8
    Wall = 9
    HQ = 10


def is_flammable(obstacle):
    return obstacle in (Obstacle.Tree, )
