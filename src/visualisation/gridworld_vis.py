import pygame

from simulation.environment import SimulationController
from simulation.obstacles import Obstacle


def render_gridworld(gridworld_controller: SimulationController, width, height):
    gridworld = gridworld_controller.model
    gridworld_width = gridworld.get_width()
    gridworld_height = gridworld.get_height()
    block_width = width / gridworld_width
    block_height = height / gridworld_height
    agent_rad = int(min(block_width, block_height) / 2)
    eye_rad = agent_rad / 3

    agent_positions = gridworld_controller.get_agent_positions()

    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))
    for y in range(gridworld_width):
        for x in range(gridworld_height):
            cell = gridworld.get_at_cell(x, y)
            colour = None
            if cell == Obstacle.OutsideMap:
                colour = (0, 0, 0)
            elif cell == Obstacle.Empty:
                colour = (255, 255, 255)

                for pos in agent_positions:
                    if abs(pos[0] - x) <= gridworld_controller.get_sight_range() and abs(pos[1] - y) <= gridworld_controller.get_sight_range():
                        colour = (220, 220, 220)
            elif cell == Obstacle.Tree:
                colour = (0, 255, 0)
            elif cell == Obstacle.Survivor:
                colour = (255, 0, 0)
            rect = pygame.Rect((x*block_width, y*block_height), (block_width+1, block_height+1))
            rect_outline = pygame.Rect((x*block_width, y*block_height), (block_width+1, block_height+1))
            pygame.draw.rect(surface, colour, rect)
            pygame.draw.rect(surface, (200, 200, 200), rect_outline, width=1)

            if (x, y) in agent_positions:
                pygame.draw.circle(
                    surface, (125, 125, 125), (int((x+0.5)*block_width), int((y+0.5)*block_height)), agent_rad)
                rotation = agent_positions[(x, y)].get_rotation()
                offset = {
                    0: (block_width/2, 0),
                    1: (block_width, block_height/2),
                    2: (block_width/2, block_height),
                    3: (0, block_height/2),
                }[rotation]
                pygame.draw.circle(
                    surface, (0, 0, 0), (int(x*block_width+offset[0]), int(y*block_height+offset[1])), eye_rad)

    return surface
