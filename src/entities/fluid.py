import random
import pygame
import numpy as np
from entities.particle import Particle
from entities.data_structures import SpatialHashGrid

class Fluid:
    """
    This class encapsulates the properties and behaviors of a fluid in a simulated environment.
    It defines the fluid characteristics and the initial setup of particles within the fluid.
    """
    def __init__(self, boundary, particle_count, damping=1.0, gravity=0, friction=1.0, target_density=1.0, pressure_coefficient=1.0, data_structure=None):
        """
        Constructor method for initialising a Fluid object with the specified properties.

        :param boundary: The rectangular boundary of the fluid environment, defined using pygame.Rect.
        :type boundary: pygame.Rect
        :param particle_count: The number of particles in the fluid.
        :type particle_count: int
        :param damping: Damping factor for the fluid motion, defaults to 1.0.
        :type damping: float, optional
        :param gravity: Gravitational force applied to the fluid, defaults to 0.
        :type gravity: float, optional
        :param friction: Friction factor in the fluid, defaults to 1.0.
        :type friction: float, optional
        :param target_density: The target density of the fluid, defaults to 1.0.
        :type target_density: float, optional
        :param pressure_coefficient: Coefficient for calculating the fluid pressure, defaults to 1.0.
        :type pressure_coefficient: float, optional
        :param data_structure: Data structure used for managing fluid particles, defaults to None.
        :type data_structure: Any, optional
        """
        self.boundary = boundary
        self.particle_count = particle_count
        self.particles = [self._create_particles() for _ in range(self.particle_count)]
        self.damping = damping
        self.gravity = gravity
        self.friction = friction
        self.target_density = target_density
        self.pressure_coefficient = pressure_coefficient

    def _create_particles(self):
        """
        Creates and initialises a single fluid particle.

        This method generates a particle with random initial positions within the specified
        boundary of the fluid environment. Each particle is created with default values for
        velocity, mass, and radius.

        The particle's initial position (x, y) is randomised to be within the right half and
        bottom half of the boundary, respectively. The particle's velocity is initially set
        to zero in both the x and y directions.

        :return: A Particle object representing the initialised fluid particle.
        :rtype: Particle

        Note: This method is intended for internal use in the fluid simulation process.
        """
        x = random.uniform(self.boundary.right / 2, self.boundary.right - 1)
        y = random.uniform(self.boundary.bottom / 2, self.boundary.bottom -1)
        x_velocity = 0.
        y_velocity = 0.
        mass = 1.0
        radius = 5.0
        return Particle(x=x, y=y, mass=mass, radius=radius, x_velocity=x_velocity, y_velocity=y_velocity)

    def draw(self, screen):
        """
        Draws the fluid particles on the given Pygame screen.

        Each particle is represented as a circle. This method goes through each particle
        in the fluid and performs two drawing steps:
        1. Draws a solid circle for the particle's core.
        2. Draws a transparent circle around the particle to create a smoothing effect.

        :param screen: The Pygame screen surface where particles will be drawn.
        :type screen: pygame.Surface

        The method uses the Pygame library's drawing functions to render circles
        and blit operations to the screen surface.
        """

        # Draws core of the particle
        for particle in self.particles:
            pygame.draw.circle(screen, (50, 50, 255), 
                               (int(particle.position[0]), int(particle.position[1])), particle.radius)


        # Draws smoothed particle radius (semi-transparent)
        alpha = 16
        for particle in self.particles:
            temp_surface = pygame.Surface((particle.smoothing_radius * 2, particle.smoothing_radius * 2), pygame.SRCALPHA)
            temp_surface.fill((0, 0, 0, 0))  # Fill with a fully transparent color
    
            pygame.draw.circle(temp_surface, (50, 50, 255, alpha), 
                           (particle.smoothing_radius, particle.smoothing_radius), 
                           particle.smoothing_radius)

            screen.blit(temp_surface, (int(particle.position[0] - particle.smoothing_radius), 
                                       int(particle.position[1] - particle.smoothing_radius)))

    def update(self, time_step):
        """
        Updates the state of each particle in the fluid for a given time step.

        This method performs several operations for each particle:
        - Resets the density and pressure.
        - Calculates the density and pressure force for each particle based on its interactions with other particles.
        - Handles particle collisions.
        - Applies pressure force, updates particle position based on the time step, and checks for boundary collisions.

        :param time_step: The time step for which the particles are updated.
        :type time_step: float
        """

        # Resets particle properties
        for i, particle in enumerate(self.particles):
            particle.density = 0
            particle.pressure = np.array([0., 0.])
            
            # Calculates particle properties based on other particles
            for j, other_particle in enumerate(self.particles):
                if i != j:
                    direction_vector, distance = particle.distance_to(other_particle)
                    particle.calculate_density(distance)
                    particle.calculate_pressure_force(other_particle, direction_vector, distance, self.target_density, self.pressure_coefficient)
                    # Handle particle collisions
                    if distance < particle.smoothing_radius + other_particle.smoothing_radius:
                        particle.collide_with(other_particle, direction_vector, distance, self.damping)
        
        # Update particle based on time step
        for particle in self.particles:
            particle.apply_pressure_force(time_step)
            particle.update_position(time_step, gravity=self.gravity)
            particle.boundary_collision(self.boundary.left, self.boundary.right, self.boundary.top, self.boundary.bottom, self.damping)

    def update_SHG(self, time_step):
        """
        Updates the state of each particle using a Spatial Hash Grid (SHG) for efficient computation.

        This method is similar to the `update` method but utilises a spatial hash grid to efficiently
        find and interact with nearby particles. It handles particle density, pressure, collisions,
        and updates positions and boundary interactions.

        :param time_step: The time step for which the particles are updated.
        :type time_step: float
        """
        # Initialise spatial hash grid
        self.SHG = SpatialHashGrid(self.boundary, self.particle_count, self.particles[0].smoothing_radius)

        # Reset particle properties and find particle neighbours
        for particle in self.particles:
            particle.density = 0
            particle.pressure = np.array([0., 0.])
            self.SHG.insert_particle(particle)
            nearby_particles = self.SHG.nearby_to(particle)

            # Calculate particle properties based on nearby particles
            for other_particle in nearby_particles:
                direction_vector, distance = particle.distance_to(other_particle)
                particle.calculate_density(distance)
                particle.calculate_pressure_force(other_particle, direction_vector, distance, self.target_density, self.pressure_coefficient)
                # Handle collisions
                if distance < particle.smoothing_radius + other_particle.smoothing_radius:
                    particle.collide_with(other_particle, direction_vector, distance, self.damping)


        # Update particle given the time step
        for particle in self.particles:
            particle.apply_pressure_force(time_step)
            particle.update_position(time_step, gravity=self.gravity)
            particle.boundary_collision(self.boundary.left, self.boundary.right, self.boundary.top, self.boundary.bottom, self.damping)

