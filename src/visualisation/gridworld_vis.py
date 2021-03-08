import pygame


def render_HUD(survivors_rescued, ts, agents_dead):
    SIZE = 20
    font = pygame.font.SysFont('arial', SIZE)

    hud_text = f"Rescued:{survivors_rescued} Ts:{ts} Agents Dead:{agents_dead}"
    text_surface = font.render(hud_text, False, (0, 0, 0))

    f_w, f_h = font.size(hud_text)
    surface = pygame.Surface((int(f_w), int(f_h)))
    surface.fill((255, 255, 255))
    surface.blit(text_surface, (0, 0))

    return surface
