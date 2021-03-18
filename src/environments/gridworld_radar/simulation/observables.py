from enum import Enum

from environments.gridworld_radar.simulation.entities import RadarDrone


class Obstacle(Enum):
    """Obstacles
    Trees can be cleared by the drones, but costs a lot of energy
    Rocks cannot be passed by ground drones, and will destroy the drone
    Road
    """
    TestObstacle = -1
    OutsideMap = 0
    Empty = 1
    Tree = 2
    BurnedTree = 3
    Rocks = 4
    Road = 5
    Wall = 6
    HQ = 7


def is_flammable(obstacle):
    return obstacle in (Obstacle.Tree, )


def is_collidable(obstacle):
    return obstacle in (Obstacle.Tree, Obstacle.BurnedTree, Obstacle.Rocks, Obstacle.Wall, Obstacle.OutsideMap)
