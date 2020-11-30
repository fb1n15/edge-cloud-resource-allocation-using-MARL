import pytest

from simulation.agent import Agent
from simulation.gridworld import GridWorldModel
from simulation.survivor import Survivor
from simulation.obstacles import OutsideMap
from simulation.obstacles import Obstacle


class TestObstacle(Obstacle):
    pass


w = 5
h = 5
s = 10


@pytest.fixture
def gridworld():
    gridworld = GridWorldModel(w, h, s)
    return gridworld


def test_survivors(gridworld):
    c = 0
    for x in range(w):
        for y in range(h):
            if isinstance(gridworld.get_at_cell(x, y), Survivor):
                c += 1
    assert c == s


def test_get_obj(gridworld):
    gridworld.set_at_cell(2, 2, TestObstacle())
    assert isinstance(gridworld.get_at_cell(2, 2), TestObstacle)


def test_bounds(gridworld):

    assert isinstance(gridworld.get_at_cell(w+1, 0), OutsideMap)
    assert isinstance(gridworld.get_at_cell(0, w+1), OutsideMap)
    assert isinstance(gridworld.get_at_cell(-1, 0), OutsideMap)
    assert isinstance(gridworld.get_at_cell(0, -1), OutsideMap)

    with pytest.raises(ValueError):
        gridworld.set_at_cell(-1, 0, None),

    with pytest.raises(ValueError):
        gridworld.set_at_cell(0, -1, None),

    with pytest.raises(ValueError):
        gridworld.set_at_cell(w+1, 0, None),

    with pytest.raises(ValueError):
        gridworld.set_at_cell(0, w+1, None),


def test_get_area():
    gridworld = GridWorldModel(w, h, 0)
    gridworld.set_at_cell(1, 1, TestObstacle())
    area = gridworld.get_area(1, 3, 1, 3)
    assert isinstance(area[0, 0], TestObstacle)
    assert area[0, 1] is None
    assert area[1, 1] is None
    assert area[1, 0] is None


def test_sight():
    gridworld = GridWorldModel(w, h, 0, world=[[TestObstacle(), None],
                                               [None, None]])
    agent = Agent(x=0, y=0, rot=0, sight=1)

    scan_area = gridworld.agent_scan(agent)
    print(scan_area)
    assert isinstance(scan_area[0, 0], TestObstacle)
    assert scan_area[1, 0] is None
    assert scan_area[0, 1] is None
    assert scan_area[1, 1] is None

    agent.rotate_right()
    scan_area2 = gridworld.agent_scan(agent)
    assert isinstance(scan_area2[0, 1], TestObstacle)
    assert scan_area2[1, 1] is None
    assert scan_area2[1, 0] is None
    assert scan_area2[0, 0] is None

    agent.rotate_right()
    scan_area3 = gridworld.agent_scan(agent)
    assert isinstance(scan_area3[1, 1], TestObstacle)
    assert scan_area3[1, 0] is None
    assert scan_area3[0, 0] is None
    assert scan_area3[0, 1] is None

    agent.rotate_right()
    scan_area4 = gridworld.agent_scan(agent)
    assert isinstance(scan_area4[1, 0], TestObstacle)
    assert scan_area4[0, 0] is None
    assert scan_area4[0, 1] is None
    assert scan_area4[1, 1] is None

    agent.rotate_right()
    scan_area5 = gridworld.agent_scan(agent)
    assert isinstance(scan_area5[0, 0], TestObstacle)
    assert scan_area5[1, 0] is None
    assert scan_area5[0, 1] is None
    assert scan_area5[1, 1] is None
