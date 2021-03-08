from environments.gridworld_obstacles.config import gridworld_obstacles_env_config
from environments.gridworld_obstacles.simulation.environment import GridWorldEnv
from environments.gridworld_obstacles.visualisation.render import render_gridworld

environments = {
    "gridworld_obstacles": {
        "env": GridWorldEnv,
        "env_config": gridworld_obstacles_env_config,
        "render": render_gridworld
    }
}
