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
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")

    # Game loop variables
    clock = pygame.time.Clock()
    time_step = 1 / 3  # 3 updates a second

    # Initialize the fluid with a certain number of particles
    num_particles = 500
    fluid = Fluid(width, height, num_particles)

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

        # Initialize SpatialHashGrid
        boundary = pygame.Rect(0, 0, width, height)
        spatial_hash_grid = SpatialHashGrid(boundary, num_particles, fluid.particles[0].radius)

        # Insert particles into SpatialHashGrid and handle collisions
        for particle in fluid.particles:
            spatial_hash_grid.insert_particle(particle)
            nearby_particles = spatial_hash_grid.get_potential_colliders(particle)
            for other in nearby_particles:
                if other != particle:
                    dx, dy, distance = particle.distance_to(other)
                    if distance < particle.radius + other.radius:
                        particle.collide_with(other, dx, dy, distance)

        # Draw the fluid particles
        fluid.draw(screen)

        # Update the display after all particles are drawn
        pygame.display.flip()


        if i > 30000:
            break

    pygame.quit()

if __name__ == '__main__':
    cProfile.run('main()', 'profile_data.prof')

