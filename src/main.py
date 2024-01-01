import cProfile
import pygame
import sys
from entities.fluid import Fluid

def main():

    pygame.init()

    width, height = 400, 400
    boundary = pygame.Rect(0, 0, width, height)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")

    clock = pygame.time.Clock()
    time_step = 1 / 3

    particle_count = 50
    fluid = Fluid(boundary, particle_count, pressure_coefficient=0.0000001, target_density=0.25, gravity=1, damping=.9)

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))

            fluid.update_SHG(time_step)

            fluid.draw(screen)

            pygame.display.flip()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        pygame.quit()

if __name__ == '__main__':
    cProfile.run('main()', 'prof.prof')
