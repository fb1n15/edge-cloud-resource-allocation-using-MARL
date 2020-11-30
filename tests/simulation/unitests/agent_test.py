from simulation.agent import Agent

from tests.simulation.unitests import TestObstacle

from unittest.mock import MagicMock


def test_sight():
    model = MagicMock()
    model.get_area.return_value = [[TestObstacle(), None],
                                   [None, None]]
    agent = Agent(x=0, y=0, rot=0, model=model, sight=0)

    scan_area = agent.scan_area()
    assert isinstance(scan_area[0, 0], TestObstacle)
    assert scan_area[1, 0] is None
    assert scan_area[0, 1] is None
    assert scan_area[1, 1] is None

    agent.rotate_right()
    scan_area2 = agent.scan_area()
    assert isinstance(scan_area2[0, 1], TestObstacle)
    assert scan_area2[1, 1] is None
    assert scan_area2[1, 0] is None
    assert scan_area2[0, 0] is None

    agent.rotate_right()
    scan_area2 = agent.scan_area()
    assert isinstance(scan_area2[1, 1], TestObstacle)
    assert scan_area2[1, 0] is None
    assert scan_area2[0, 0] is None
    assert scan_area2[0, 1] is None

    agent.rotate_right()
    scan_area2 = agent.scan_area()
    assert isinstance(scan_area2[1, 0], TestObstacle)
    assert scan_area2[0, 0] is None
    assert scan_area2[0, 1] is None
    assert scan_area2[1, 1] is None

    agent.rotate_right()
    scan_area2 = agent.scan_area()
    assert isinstance(scan_area[0, 0], TestObstacle)
    assert scan_area[1, 0] is None
    assert scan_area[0, 1] is None
    assert scan_area[1, 1] is None
