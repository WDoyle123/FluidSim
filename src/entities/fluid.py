import random
import pygame
from entities.particle import Particle

class Fluid:
    """
    Represents a fluid composed of many particles.

    Attributes:
        particles (list of Particle): The particles that make up the fluid.
        width (int): Width of the area in which the fluid is contained.
        height (int): Height of the area in which the fluid is contained.
    """
    def __init__(self, width, height, num_particles, restitution = 1.0, gravity = 0, friction = 1.0):
        """
        Initialise the fluid with a set number of particles.

        Args:
            width (int): Width of the area in which the fluid is contained.
            height (int): Height of the area in which the fluid is contained.
            num_particles (int): Number of particles in the fluid.
        """
        self.width = width
        self.height = height
        self.particles = [self._create_particle() for _ in range(num_particles)]
        self.restitution = restitution
        self.gravity = gravity
        self.friction = friction
        self.density = num_particles / (width * height)

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
        for particle in self.particles:
            particle.update_position(time_step, gravity=self.gravity, friction=self.friction)
            particle.boundary_collision(0, self.height, 0, self.width, restitution=self.restitution)

    def calculate_density(self):
        """
        Calculate the density of the fluid in different regions.
        """
        pass  # Implement density calculation

    def draw(self, screen):
        """
        Draw the fluid particles on the screen.

        Args:
            screen (pygame.Surface): The pygame surface to draw the fluid on.
        """
        for particle in self.particles:
            pygame.draw.circle(screen, (0, 0, 255), 
                               (int(particle.x_position), int(particle.y_position)), 
                               particle.radius)

