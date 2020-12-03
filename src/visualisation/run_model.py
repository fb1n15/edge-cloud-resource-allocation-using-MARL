from time import sleep

import pygame
import thorpy
import threading

from simulation.environment import SimulationController, GridWorldEnv
from simulation.gridworld import GridWorldModel
from learning import training
from learning.training import run_same_policy, load_checkpoints
from visualisation.gridworld_vis import render_gridworld
import ray.rllib.agents.ppo as ppo


# TODO Move this stuff out of here
checkpoints = load_checkpoints()
agent = ppo.PPOTrainer(config=training.config, env=GridWorldEnv)
agent.restore(checkpoints[-1][0])  # Restore the last checkpoint
env = GridWorldEnv(training.config["env_config"])
gridworld = env.controller
# gridworld = SimulationController(20, 20, 10, 2, 4, 200)
# gridworld.initialise()

print(checkpoints[-1])

episode_reward = 0
done = False
obs = env.reset()
running = True


def step_simulation():
    global done, obs
    while running:
        sleep(0.2)
        action = {}
        for agent_id, agent_obs in obs.items():
            action[agent_id] = agent.compute_action(agent_obs)
        obs, reward, done, info = env.step(action)
        # episode_reward += reward


def restart_simulation():
    global done, obs
    done = False
    obs = env.reset()


WIDTH = 640
HEIGHT = 720

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
button = thorpy.make_button("Restart", func=restart_simulation)
box = thorpy.Box(elements=[slider, button])
menu = thorpy.Menu(box)

# Set the screen as surface for all elements
for element in menu.get_population():
    element.surface = screen

box.set_topleft((0, 0))
box.blit()
box.update()

t = threading.Thread(target=step_simulation)
t.start()

running = True
while running:
    clock.tick(60)
    screen.blit(render_gridworld(gridworld, WIDTH, HEIGHT-80), (0, 80))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        menu.react(event)
    pygame.display.update()

pygame.quit()
running = False