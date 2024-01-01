import cProfile
import pygame
import sys
from entities.fluid import Fluid

def main():

    pygame.init()

    width, height = 600, 600
    boundary = pygame.Rect(0, 0, width, height)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")

    clock = pygame.time.Clock()
    time_step = 1 / 3

    particle_count = 250
    fluid = Fluid(boundary, particle_count, pressure_coefficient=0.0000001, target_density=0.25, gravity=0, damping=.9)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        screen.fill((0, 0, 0))

        fluid.update_SHG(time_step)

        fluid.draw(screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    cProfile.run('main()', 'prof.prof')
