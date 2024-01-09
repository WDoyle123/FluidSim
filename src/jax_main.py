import cProfile
import pygame
import sys
from entities.jax_fluid import Fluid

def main():

    pygame.init()

    width, height = 400, 400
    boundary = pygame.Rect(0, 0, width, height)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")

    clock = pygame.time.Clock()
    time_step = 1 / 3

    particle_count = 10
    fluid = Fluid(boundary, particle_count, pressure_coefficient=0.0000001, target_density=0.25, gravity=0.1, damping=.9)
    counter = 0 
    target = 300

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))

            fluid.update_fluid(time_step)

            fluid.draw(screen)

            counter += 1
            if counter == target:
                running = False

            pygame.display.flip()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        pygame.quit()

if __name__ == '__main__':
    cProfile.run('main()', 'prof.prof')
