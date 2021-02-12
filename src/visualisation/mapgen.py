from visualisation import gridworld_vis
import pygame
from simulation.environment import GridWorldEnv
from common.config import env_config
from visualisation.run_model import WIDTH, HEIGHT


def main():
    """This is for testing the map generation and graphics"""
    running = True
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    env_controller = GridWorldEnv(env_config).controller
    env_controller.initialise()
    screen.blit(gridworld_vis.render_gridworld(env_controller, WIDTH, HEIGHT - 80), (0, 80))
    pygame.display.update()
    clock = pygame.time.Clock()
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
