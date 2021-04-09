from environments.gridworld_obstacles.simulation.environment import GridWorldObstaclesEnv
from environments.gridworld_radar.simulation.environment import GridWorldRadarRescueEnv


def environment_map(name):
    environments = {
        "gridworld_obstacles_vision_net": {
            "env": GridWorldObstaclesEnv,
            "render": GridWorldObstaclesEnv.render_method()
        },
        "gridworld_radar_vision_net": {
            "env": GridWorldRadarRescueEnv,
            "render": GridWorldRadarRescueEnv.render_method()
        }
    }
    return environments[name]
