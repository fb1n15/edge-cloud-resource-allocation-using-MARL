from environments.gridworld_obstacles.simulation.environment import GridWorldObstaclesEnv
from environments.gridworld_radar.simulation.environment import GridWorldRadarRescueEnv
from environments.edge_cloud.simulation.environment import EdgeCloudEnv, EdgeCloudEnv1


def environment_map(name):
    environments = {
        "edge_cloud": {
            "env": EdgeCloudEnv,
            "render": EdgeCloudEnv.render_method()
        },
        "edge_cloud1": {
            "env": EdgeCloudEnv1,
            "render": EdgeCloudEnv1.render_method()
            },
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
