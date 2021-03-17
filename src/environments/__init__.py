from environments.gridworld_obstacles.simulation.environment import GridWorldEnv
from environments.gridworld_obstacles.visualisation.render import render_gridworld

from environments.gridworld_radar.simulation.environment import GridWorldEnv
from environments.gridworld_radar.visualisation.render import render_gridworld


def environment_map(name):
    environments = {
        "gridworld_obstacles_vision_net": {
            "env": GridWorldEnv,
            "render": render_gridworld
        },
        "gridworld_radar_vision_net": {
            "env": GridWorldEnv,
            "render": render_gridworld
        }
    }
    return environments[name]
