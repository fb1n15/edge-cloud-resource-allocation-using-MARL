import pygame
import ray
import thorpy
import threading

from gym.spaces import Tuple
from ray.tune import register_env

from environments import environment_map
from learning.training import CustomCallbacks, get_trainer_config
from visualisation.gridworld_vis import render_HUD
import ray.rllib.agents.ppo as ppo

WIDTH = 640
HEIGHT = 720


def training_config(config):
    trainer_config = config["trainer-config"]
    trainer_config["env_config"] = config["env-config"]

    # Choose environment, with groupings
    env = environment_map(config["env"])["env"]
    if "grouping" not in config:
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, env.get_observation_space(config["env-config"]), env.get_action_space(), {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }

    elif config["grouping"] == "all_same":
        obs_space = Tuple(
            [env.get_observation_space(config["env-config"]) for i in range(config["env-config"]["num_agents"])])
        act_space = Tuple([env.get_action_space() for i in range(config["env-config"]["num_agents"])])
        grouping = {
            "group_1": ["drone_" + str(i) for i in range(config["env-config"]["num_agents"])],
        }

        # Register the environment with Ray, and use this in the config
        register_env(config["env"],
                     lambda env_cfg: env(env_cfg).with_agent_groups(grouping, obs_space=obs_space, act_space=act_space))
        trainer_config["env"] = config["env"]
        # trainer_config["env"] = env

        # trainer_config["multiagent"] = {
        #     "policies": {
        #         "default": (None, env.get_observation_space(config["env-config"]), env.get_action_space(), {}),
        #     },
        #     "policy_mapping_fn": lambda agent_id: "default"
        # }

        trainer_config["multiagent"] = {
            "policies": {
                "default": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "default"
        }

    elif config["grouping"] == "radar-rescue":
        trainer_config["env"] = env

        trainer_config["multiagent"] = {
            "policies": {
                "radar": (
                None, env.get_observation_space(config["env-config"], "radar"), env.get_action_space("radar"), {}),
                "rescue": (
                None, env.get_observation_space(config["env-config"], "rescue"), env.get_action_space("rescue"), {}),
            },
            "policy_mapping_fn": lambda agent_id: agent_id.split("_")[0]
        }

    # ModelCatalog.register_custom_model("CustomVisionNetwork", CustomVisionNetwork)

    # Add callbacks for custom metrics
    trainer_config["callbacks"] = CustomCallbacks

    return trainer_config


class SimulationRunner:
    def __init__(self, experiment, env, config):

        # Create logger which doesn't do anything
        # del experiment["best trial"]["config"]["callbacks"]  # Get rid of any callbacks
        # experiment["best trial"]["config"]["explore"] = False
        # # TODO remove sampled stuff from config
        # trainer_config = {"env_config": training_config(config)["env_config"],
        #                   "multiagent": training_config(config)["multiagent"],
        #                   "model": training_config(config)["model"],
        #                   "framework": "torch",
        #                   "explore": False}
        self.env = env(config["env-config"])
        trainer_config = get_trainer_config(config)
        self.agent = ppo.PPOTrainer(config=trainer_config,
                                    env=env)
        # path = r"C:\Users\Jack\PycharmProjects\marl-disaster-relief\src\results\DroneRescue DroneRescue gridworld_radar_vision_net_ppo\PPO_GridWorldEnv_e712e_00000_0_lambda=0.9112,lr=2.8076e-05_2021-03-22_17-34-49\checkpoint_000200\checkpoint-200"
        self.agent.restore(experiment)  # Restore the last checkpoint
        # self.agent.restore(experiment["best trial"]["path"])  # Restore the last checkpoint

        self.gridworld = self.env.controller

        self.episode_reward = 0
        self.done = False
        self.obs = self.env.reset()
        self.running = True

        self.speed_callback = None
        self.timestep = 0

        policy_map = self.agent.workers.local_worker().policy_map
        self.model_state = {p: m.get_initial_state() for p, m in policy_map.items()}
        print(self.model_state)

    def set_speed_callback(self, callback):
        self.speed_callback = callback

    def step_simulation(self):
        clock = pygame.time.Clock()
        while self.running:
            self.timestep += 1
            fr = self.speed_callback()
            if fr is not None:
                clock.tick(self.speed_callback())
            else:
                raise Exception("speed callback not set")
            action = {}
            for agent_id, agent_obs in self.obs.items():
                policy_id = agent_id.split("_")[0]
                if policy_id not in self.model_state:
                    policy_id = "default"
                action[agent_id], self.model_state[policy_id], _ = self.agent.compute_action(
                    observation=agent_obs,
                    policy_id=policy_id,
                    state=self.model_state[policy_id],
                    full_fetch=True
                )
            self.obs, self.reward, self.done, self.info = self.env.step(action)
            if self.done["__all__"]:
                self.restart_simulation()
        # episode_reward += reward
        # Check if all the drones have run out of battery

    def restart_simulation(self):
        self.done = False
        self.obs = self.env.reset()
        self.timestep = 0

    def get_time(self):
        return self.timestep

    def get_rescued(self):
        return self.env.get_survivors_rescued()

    def num_agents_crashed(self):
        return self.env.num_agents_crashed()


def start_displaying(runner, env):
    MENU_HEIGHT = 80

    pygame.init()
    pygame.font.init()
    pygame.key.set_repeat(300, 30)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill((255, 255, 255))
    rect = pygame.Rect((0, 0, 50, 50))
    rect.center = screen.get_rect().center
    clock = pygame.time.Clock()

    pygame.draw.rect(screen, (255, 0, 0), rect)
    pygame.display.flip()

    # Declaration of some ThorPy elements ...
    slider = thorpy.SliderX(100, (0.1, 15), "Speed")
    slider.set_value(5)
    runner.set_speed_callback(lambda: slider.get_value())
    button = thorpy.make_button("Restart", func=runner.restart_simulation)
    box = thorpy.Box(elements=[slider, button])
    menu = thorpy.Menu(box)

    # Set the screen as surface for all elements
    for element in menu.get_population():
        element.surface = screen

    box.set_topleft((0, 0))
    box.blit()
    box.update()

    t = threading.Thread(target=runner.step_simulation)
    t.start()

    running = True
    while running:
        clock.tick(60)

        screen.blit(env["render"](runner.gridworld, WIDTH, HEIGHT - MENU_HEIGHT), (0, MENU_HEIGHT))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            menu.react(event)
        hud = render_HUD(runner.get_rescued(), runner.get_time(), runner.num_agents_crashed())
        screen.blit(hud, (WIDTH-hud.get_width(), 0))
        pygame.display.update()

    pygame.quit()
    runner.running = False


def main(experiment, config):
    ray.init()
    env = environment_map(config["env"])
    runner = SimulationRunner(experiment, env["env"], config)
    start_displaying(runner, env)
