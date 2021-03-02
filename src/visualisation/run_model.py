from time import sleep

import pygame
import ray
import thorpy
import threading

from simulation.environment import GridWorldEnv
from visualisation.gridworld_vis import render_gridworld
import ray.rllib.agents.ppo as ppo

WIDTH = 640
HEIGHT = 720


class SimulationRunner:
    def __init__(self, experiment):

        # Create logger which doesn't do anything
        self.agent = ppo.PPOTrainer(config=experiment["best trial"]["config"],
                                    env=GridWorldEnv)
        self.agent.restore(experiment["best trial"]["path"])  # Restore the last checkpoint
        self.env = GridWorldEnv(experiment["environment"])

        self.gridworld = self.env.controller

        self.episode_reward = 0
        self.done = False
        self.obs = self.env.reset()
        self.running = True

    def step_simulation(self):
        clock = pygame.time.Clock()
        while self.running:
            clock.tick(5)
            action = {}
            for agent_id, agent_obs in self.obs.items():
                action[agent_id] = self.agent.compute_action(agent_obs)
            self.obs, self.reward, self.done, self.info = self.env.step(action)
        # episode_reward += reward
        # Check if all the drones have run out of battery

    def restart_simulation(self):
        self.done = False
        self.obs = self.env.reset()


def start_displaying(runner):
    pygame.init()
    pygame.key.set_repeat(300, 30)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill((255, 255, 255))
    rect = pygame.Rect((0, 0, 50, 50))
    rect.center = screen.get_rect().center
    clock = pygame.time.Clock()

    pygame.draw.rect(screen, (255, 0, 0), rect)
    pygame.display.flip()

    # Declaration of some ThorPy elements ...
    slider = thorpy.SliderX(100, (12, 35), "Speed")
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

        # runner.step_simulation()

        screen.blit(render_gridworld(runner.gridworld, WIDTH, HEIGHT - 80), (0, 80))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            menu.react(event)
        pygame.display.update()

    pygame.quit()
    runner.running = False


def main(experiment):
    ray.init()
    runner = SimulationRunner(experiment)
    start_displaying(runner)
