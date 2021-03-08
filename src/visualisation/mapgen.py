import thorpy

import pygame

from environments import environments
from visualisation import gridworld_vis
from visualisation.run_model import WIDTH, HEIGHT


def main(env=environments["gridworld_obstacles"]):
    """This is for testing the map generation and graphics"""
    env_controller = env["env"](env["env_config"]).controller
    env_controller.initialise()

    pygame.init()
    running = True
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Declaration of some ThorPy elements ...
    button = thorpy.make_button("Reset", func=env_controller.initialise)
    box = thorpy.Box(elements=[button])
    menu = thorpy.Menu(box)

    # Set the screen as surface for all elements
    for element in menu.get_population():
        element.surface = screen

    box.set_topleft((0, 0))
    box.blit()
    box.update()

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            menu.react(event)

        screen.blit(env["render"](env_controller, WIDTH, HEIGHT - 80), (0, 80))
        pygame.display.update()
