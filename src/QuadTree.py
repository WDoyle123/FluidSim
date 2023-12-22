import cProfile

import pygame
import sys
from entities.particle import Particle
from entities.data_structures import QuadTree
import random

def main():

    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 600, 600
    floor, ceiling = 0, height
    left, right = 0, width

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")

    # Game loop variables
    clock = pygame.time.Clock()
    time_step = 1 / 3 # Update the simulation by 1/60th of a second each frame

    # Number of particles to generate
    num_particles = 500

    # Create multiple particle instances with random properties
    particles = []
    for _ in range(num_particles):
        x = random.randrange(left + 10, right - 10) 
        y = random.randrange(int(floor), int(height / 2))
        radius = 5
        x_velocity = random.uniform(-50, 50)  
        y_velocity = random.uniform(-0, 0)  
        particle = Particle(x, y, radius=radius, x_velocity=x_velocity, y_velocity=y_velocity)
        particles.append(particle)

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

        boundary = pygame.Rect(0, 0, width, height)
        quadtree = QuadTree(boundary, capacity=16)

        for particle in particles:
            # Update the particle
            particle.update_position(time_step, ceiling=ceiling)
            particle.boundary_collision(floor, ceiling, left, right)
            quadtree.insert(particle)

        #quadtree.draw(screen)

        for particle in particles:

            # Check for collisions
            collision_box = pygame.Rect(particle.x_position - particle.radius,
                                particle.y_position - particle.radius,
                                particle.radius * 2, particle.radius * 2)

            nearby_particles = []
            quadtree.query(collision_box, nearby_particles)

            for other in nearby_particles:
                if other != particle:
                    dx, dy, distance = particle.distance_to(other)
                    if distance < particle.radius + other.radius:
                        particle.collide_with(other, dx, dy, distance)

            # Draw the particle after updating its position
            pygame.draw.circle(screen, (0, 0, 255), 
                (int(particle.x_position), int(particle.y_position)), 
                particle.radius)
    
        # Update the display after all particles are drawn
        pygame.display.flip()
        clock.tick(60)

        if i > 300:
            break

    pygame.quit()

if __name__ == '__main__':
    cProfile.run('main()', 'quad_tree.prof')
