import cProfile
import pygame
import sys
from entities.particle import Particle
from entities.fluid import Fluid
from entities.data_structures import SpatialHashGrid

def main():

    # Initialise Pygame
    pygame.init()

    # Set up the display
    width, height = 600, 600
    boundary = pygame.Rect(0, 0, width, height)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")

    # Game loop variables
    clock = pygame.time.Clock()
    time_step = 1 / 3  # 3 updates a second

    # Initialize the fluid with a certain number of particles
    num_particles = 500
    fluid = Fluid(boundary, num_particles)

    i = 0

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        i += 1

        # Clear the screen at the beginning of each frame
        screen.fill((0, 0, 0))

        # Update the fluid
        fluid.update(time_step)

        # Draw the fluid particles
        fluid.draw(screen)

        # Update the display after all particles are drawn
        pygame.display.flip()


        if i > 30000:
            break

    pygame.quit()

if __name__ == '__main__':
    cProfile.run('main()', 'profile_data.prof')

