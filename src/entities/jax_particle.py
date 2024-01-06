import jax.numpy as jnp
from jax import jit, grad

from calculations.kernels import jax_spiky_smoothing_kernel_2d, jax_spiky_smoothing_kernel_2d_derivative

class Particle:
    def __init__(self, position, velocity, radius=5, pressure=None):
        self.position = jnp.array(position)
        self.velocity = jnp.array(velocity)
        self.mass = 1
        self.radius = radius
        self.smoothing_radius = radius * 4
        self.density = 0
        self.pressure = jnp.array(pressure) if pressure is not None else jnp.array([0., 0.])

    def step(self, time_step, gravity=0, left=0, right=100, top=0, bottom=100, damping=1.0):
        new_velocity = self.apply_pressure_force(self.velocity, self.pressure, self.density, time_step)
        new_position = self.update_position(self.position, new_velocity, time_step, gravity)
        new_position, new_velocity = self.boundary_collision(
            new_position, new_velocity, self.radius, left, right, top, bottom, damping)

        return Particle(new_position, new_velocity, self.radius)

    def distance_to(self, other):
        delta = self.position - other.position
        distance = jnp.sqrt(jnp.sum(delta ** 2))

        epsilon = 1e-8
        direction_vector = delta / (distance + epsilon)
        return direction_vector, distance

    @staticmethod
    @jit
    def update_position(position, velocity, time_step, gravity=0):
        new_velocity = velocity.at[1].add(gravity * time_step)
        new_position = position + new_velocity * time_step
        return new_position

    @staticmethod
    @jit
    def apply_pressure_force(velocity, pressure, density, time_step):
        pressure_acceleration = pressure / (density + 1e-8)
        new_velocity = velocity + pressure_acceleration * time_step
        return new_velocity

    @staticmethod
    @jit
    def calculate_density(distance, smoothing_radius, density, mass):
        influence = jax_spiky_smoothing_kernel_2d(distance, smoothing_radius)
        new_density = density + (mass * influence)
        return new_density


    @staticmethod
    @jit
    def convert_density_to_pressure(density, target_density, pressure_coefficient):
        density_error = density - target_density
        pressure = density_error * pressure_coefficient
        return pressure

    @staticmethod
    @jit
    def calculate_pressure_force(pressure, own_mass, other_mass, density, direction_vector,
                                 distance, target_density, pressure_coefficient, smoothing_radius):

        gradient_magnitude = jax_spiky_smoothing_kernel_2d_derivative(distance, smoothing_radius)
        pressure_force_magnitude = (- other_mass * 
                                    (own_mass * target_density * pressure_coefficient) /
                                    (density + 1e-8) *
                                    gradient_magnitude)

        pressure_force_vector = pressure_force_magnitude * direction_vector
        new_pressure = pressure + pressure_force_vector
        return new_pressure

    @staticmethod
    @jit
    def collide_with(velocity_self, velocity_other, position_self, position_other,
                     mass_self, mass_other, smoothing_radius_self, smoothing_radius_other,
                     direction_vector, distance, positional_correction_factor=0.001, damping=1.0):
        
        relative_velocity = velocity_self - velocity_other
        velocity_along_normal = jnp.dot(relative_velocity, direction_vector)

        impulse = -(1 + damping) * velocity_along_normal / (1 / mass_self + 1 / mass_other)
        impulse_vector = impulse * direction_vector

        new_velocity_self = velocity_self + impulse_vector / mass_self
        new_velocity_other = velocity_other - impulse_vector / mass_other
            
        correction_magnitude = positional_correction_factor * jnp.maximum(0, smoothing_radius_self + smoothing_radius_other - distance)
        correction_vector = correction_magnitude * direction_vector
        new_position_self = position_self + correction_vector / 2
        new_position_other = position_other - correction_vector / 2

        return new_velocity_self, new_velocity_other, new_position_self, new_position_other

        #return velocity_self, velocity_other, position_self, position_other

    @staticmethod
    @jit
    def boundary_collision(position, velocity, radius, left, right, top, bottom, damping=1.0):
        new_position = position
        new_velocity = velocity

        # Collision with left and right walls
        new_position = jnp.where(new_position[0] - radius <= left, left + radius, new_position)
        new_velocity = jnp.where(new_position[0] - radius <= left, -new_velocity[0] * damping, new_velocity)

        new_position = jnp.where(new_position[0] + radius >= right, right - radius, new_position)
        new_velocity = jnp.where(new_position[0] + radius >= right, -new_velocity[0] * damping, new_velocity)

        # Collision with top and bottom walls
        new_position = jnp.where(new_position[1] - radius <= top, top + radius, new_position)
        new_velocity = jnp.where(new_position[1] - radius <= top, -new_velocity[1] * damping, new_velocity)

        new_position = jnp.where(new_position[1] + radius >= bottom, bottom - radius, new_position)
        new_velocity = jnp.where(new_position[1] + radius >= bottom, -new_velocity[1] * damping, new_velocity)

        return new_position, new_velocity
