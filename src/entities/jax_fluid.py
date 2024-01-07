import pygame

import jax.numpy as jnp
from jax import jit, random, vmap

from entities.jax_particle import Particle
from calculations.distance import distance_to

class Fluid:
    def __init__(self, boundary, particle_count, damping=1.0, gravity=0., friction=0.9, target_density=1.0, pressure_coefficient=1.0, positional_correction_factor=0.001):
        self.boundary = boundary
        self.left, self.top, self.right, self.bottom = self.boundary
        self.particle_count = particle_count
        self.damping = damping
        self.gravity = gravity
        self.friction = friction
        self.target_density = target_density
        self.pressure_coefficient = pressure_coefficient
        self.positional_correction_factor = positional_correction_factor
        self.key = random.PRNGKey(0)
        self.particles = [self._create_particle() for _ in range(particle_count)]

    def _create_particle(self):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey, minval=self.boundary.right / 2, maxval=self.boundary.right - 1)
        y = random.uniform(self.key, minval=self.boundary.bottom / 2, maxval=self.boundary.bottom - 1)
        position = jnp.array([x, y])
        velocity = jnp.array([2., 0.])
        radius = 5.0
        return Particle(position=position, velocity=velocity, radius=radius)

    def draw(self, screen):
        for particle in self.particles:
            pygame.draw.circle(screen, (50, 50, 255),
                               (int(particle.position[0]), int(particle.position[1])), particle.radius)

    def update_fluid(self, time_step):
        particle_positions = jnp.array([p.position for p in self.particles])
        particle_velocities = jnp.array([p.velocity for p in self.particles])

        # Reset values
        particle_densities = jnp.array([p.density for p in self.particles]) * 0
        particle_pressures = jnp.array([p.pressure for p in self.particles]) * 0

        state = (time_step, self.gravity, self.damping, self.particles[0].radius, self.left, self.top, self.right, self.bottom)
        
        # Vectorise update_particle function
        vectorised_update_particles = vmap(self.update_particle, in_axes=(0, 0, None))
        new_positions, new_velocities = vectorised_update_particles(particle_positions, particle_velocities, state)

        # Make new particles
        self.particles = [Particle(pos, vel) for pos, vel in zip(new_positions, new_velocities)]


    @staticmethod
    def update_particle(position, velocity, state):
        time_step, gravity, damping, radius, left, top, right, bottom = state
    
        update_position, update_velocity = Particle.update_position(position, velocity, time_step, gravity)
        new_position, new_velocity = Particle.left_right_boundary_collision(update_position, update_velocity,
                                                                            radius, left, right, damping)
        new_position, new_velocity = Particle.top_bottom_boundary_collision(new_position, new_velocity,
                                                                            radius, top, bottom, damping)

        return new_position, new_velocity

    def update_old(self, time_step):

        # Make list to store new particles
        new_particles = []

        # Store intermediate collision results
        collision_updates = {}

        # Loop through particles and reset values
        for i, particle in enumerate(self.particles):
            new_density = 0
            new_pressure = jnp.array([0., 0.])

            # Loop through other particles relative to a particle
            for j, other_particle in enumerate(self.particles):
                if i != j:

                    # Calculate distance to other particle
                    direction_vector, distance = distance_to(particle.position, other_particle.position)

                    # Calculate resultant density and pressure
                    new_density = Particle.calculate_density(new_density, particle.mass, distance, particle.smoothing_radius)
                    new_pressure = Particle.calculate_pressure_force(
                        new_pressure, particle.mass, other_particle.mass, other_particle.density,
                        direction_vector, distance, self.target_density, self.pressure_coefficient, particle.smoothing_radius
                    )
                    """
                    # Collision check
                    if distance < particle.smoothing_radius + other_particle.smoothing_radius:
                        # Only process collision once per pair
                        if (j, i) not in collision_updates:
                            new_vel_self, new_vel_other, new_pos_self, new_pos_other = Particle.collide_with(
                                particle.velocity, other_particle.velocity, particle.position, other_particle.position,
                                particle.mass, other_particle.mass, particle.smoothing_radius, other_particle.smoothing_radius,
                                direction_vector, distance, self.positional_correction_factor, self.damping
                            )
                            collision_updates[(i, j)] = (new_vel_self, new_pos_self)
                            collision_updates[(j, i)] = (new_vel_other, new_pos_other)
                     """
            # Retrieve updated velocity and position from collision handling
            if i in collision_updates:
                new_velocity, new_position = collision_updates[i]

            # Apply pressure force, update position, handle boundary collisions
            new_velocity = Particle.apply_pressure_force(particle.velocity, new_pressure, particle.density, time_step)
            new_position = Particle.update_position(particle.position, new_velocity, time_step, self.gravity)
            """
            new_position, new_velocity = Particle.boundary_collision(
            new_position, new_velocity, particle.radius,
            self.boundary.left, self.boundary.right, self.boundary.top, self.boundary.bottom, self.damping
            )
            """
            # Create a new Particle instance with updated properties
            new_particles.append(Particle(new_position, new_velocity, particle.radius))

        self.particles = new_particles
