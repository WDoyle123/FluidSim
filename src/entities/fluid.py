import random
import pygame
from entities.particle import Particle
from entities.data_structures import SpatialHashGrid

class Fluid:
    """
    Represents a fluid composed of many particles.

    Attributes:
        particles (list of Particle): The particles that make up the fluid.
        width (int): Width of the area in which the fluid is contained.
        height (int): Height of the area in which the fluid is contained.
    """
    def __init__(self, boundary, num_particles, restitution = 1.0, gravity = 0, friction = 1.0):
        """
        Initialise the fluid with a set number of particles.

        Args:
            width (int): Width of the area in which the fluid is contained.
            height (int): Height of the area in which the fluid is contained.
            num_particles (int): Number of particles in the fluid.
        """
        self.boundary = boundary
        _, _, self.width, self.height = self.boundary
        self.num_particles = num_particles
        self.particles = [self._create_particle() for _ in range(self.num_particles)]
        self.restitution = restitution
        self.gravity = gravity
        self.friction = friction
        self.SHG = SpatialHashGrid(self.boundary, num_particles, self.particles[0].radius)

    def _create_particle(self):
        """
        Create a single particle with random properties.

        Returns:
            Particle: A new particle instance.
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        mass = 1.0
        radius = 5.0
        x_velocity = random.uniform(-1, 1)
        y_velocity = random.uniform(-1, 1)
        return Particle(x, y, mass, radius, x_velocity, y_velocity)

    def update(self, time_step):
        """
        Update the state of the fluid for the next frame.

        Args:
            time_step (float): The time step for the update.
        """
        self.SHG = SpatialHashGrid(self.boundary, self.num_particles, self.particles[0].radius)
        for particle in self.particles:
            particle.update_position(time_step, gravity=self.gravity, friction=self.friction)
            particle.boundary_collision(0, self.height, 0, self.width, restitution=self.restitution)

            # Insert particles into SpatialHashGrid and handle collisions
            self.SHG.insert_particle(particle)
            nearby_particles = self.SHG.get_potential_colliders(particle)
            for other in nearby_particles:
                dx, dy, distance = particle.distance_to(other)
                particle.calculate_density_derivative(other)
                print(particle.x_density_gradient, particle.y_density_gradient)
                if distance < particle.radius + other.radius:
                   particle.collide_with(other, dx, dy, distance)


    def draw(self, screen):
        """
        Draw the fluid particles on the screen.
    
        Args:
            screen (pygame.Surface): The pygame surface to draw the fluid on.
        """

        # Then draw the smaller blue circles on top without transparency
        for particle in self.particles:
            pygame.draw.circle(screen, (50, 50, 255), 
                               (int(particle.x_position), int(particle.y_position)), 
                               particle.radius)

        red_alpha = 64  # 25% transparancy

        for particle in self.particles:
            temp_surface = pygame.Surface((particle.radius * 10, particle.radius * 10), pygame.SRCALPHA)
            temp_surface.fill((0, 0, 0, 0))
            pygame.draw.circle(temp_surface, (50, 50, 255, red_alpha), 
                               (particle.radius * 5, particle.radius * 5), 
                               particle.radius * 5)

            screen.blit(temp_surface, (int(particle.x_position - particle.radius * 5), 
                                       int(particle.y_position - particle.radius * 5)))



