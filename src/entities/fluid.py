import random
import pygame
import numpy as np
from entities.particle import Particle
from entities.data_structures import SpatialHashGrid

class Fluid:
    def __init__(self, boundary, particle_count, damping=1.0, gravity=0, friction=1.0, target_density=1.0, pressure_coefficient=1.0, data_structure=None):
        self.boundary = boundary
        self.particle_count = particle_count
        self.particles = [self._create_particles() for _ in range(self.particle_count)]
        self.damping = damping
        self.gravity = gravity
        self.friction = friction
        self.target_density = target_density
        self.pressure_coefficient = pressure_coefficient

    def _create_particles(self):
        x = random.uniform(self.boundary.right / 2, self.boundary.right - 1)
        y = random.uniform(self.boundary.bottom / 2, self.boundary.bottom -1)
        x_velocity = 0.
        y_velocity = 0.
        mass = 1.0
        radius = 5.0
        return Particle(x=x, y=y, mass=mass, radius=radius, x_velocity=x_velocity, y_velocity=y_velocity)

    def draw(self, screen):
        for particle in self.particles:
            pygame.draw.circle(screen, (50, 50, 255), 
                               (int(particle.position[0]), int(particle.position[1])), particle.radius)

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

        for i, particle in enumerate(self.particles):
            particle.density = 0
            particle.pressure = np.array([0., 0.])
            
            for j, other_particle in enumerate(self.particles):
                if i != j:
                    direction_vector, distance = particle.distance_to(other_particle)
                    particle.calculate_density(distance)
                    particle.calculate_pressure_force(other_particle, direction_vector, distance, self.target_density, self.pressure_coefficient)
                    if distance < particle.smoothing_radius + other_particle.smoothing_radius:
                        particle.collide_with(other_particle, direction_vector, distance, self.damping)

        for particle in self.particles:
            particle.apply_pressure_force(time_step)
            particle.update_position(time_step, gravity=self.gravity)
            particle.boundary_collision(self.boundary.left, self.boundary.right, self.boundary.top, self.boundary.bottom, self.damping)

    def update_SHG(self, time_step):

        self.SHG = SpatialHashGrid(self.boundary, self.particle_count, self.particles[0].smoothing_radius)

        for particle in self.particles:
            particle.density = 0
            particle.pressure = np.array([0., 0.])
            self.SHG.insert_particle(particle)
            nearby_particles = self.SHG.nearby_to(particle)

            for other_particle in nearby_particles:
                direction_vector, distance = particle.distance_to(other_particle)
                particle.calculate_density(distance)
                particle.calculate_pressure_force(other_particle, direction_vector, distance, self.target_density, self.pressure_coefficient)
                if distance < particle.smoothing_radius + other_particle.smoothing_radius:
                    particle.collide_with(other_particle, direction_vector, distance, self.damping)


        for particle in self.particles:
            particle.apply_pressure_force(time_step)
            particle.update_position(time_step, gravity=self.gravity)
            particle.boundary_collision(self.boundary.left, self.boundary.right, self.boundary.top, self.boundary.bottom, self.damping)






