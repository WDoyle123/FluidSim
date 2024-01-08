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


    @staticmethod
    def update_position(position, velocity, time_step, gravity=0):
        new_position = position
        new_velocity = velocity
        
        new_velocity = velocity.at[1].add(gravity * time_step)
        new_position = position + new_velocity * time_step

        return new_position, new_velocity

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
    def left_right_boundary_collision(position, velocity, radius, left, right, damping):
        collision_left = position[0] - radius <= left
        collision_right = position[0] + radius >= right

        new_x = jnp.where(collision_left, left + radius, position[0])
        new_x = jnp.where(collision_right, right - radius, new_x)
        new_velocity_x = jnp.where(collision_left | collision_right, -velocity[0] * damping, velocity[0])

        return jnp.array([new_x, position[1]]), velocity.at[0].set(new_velocity_x)

    @staticmethod
    @jit
    def top_bottom_boundary_collision(position, velocity, radius, top, bottom, damping):
        collision_top = position[1] - radius <= top
        collision_bottom = position[1] + radius >= bottom

        new_y = jnp.where(collision_top, top + radius, position[1])
        new_y = jnp.where(collision_bottom, bottom - radius, new_y)
        new_velocity_y = jnp.where(collision_top | collision_bottom, -velocity[1] * damping, velocity[1])

        return jnp.array([position[0], new_y]), velocity.at[1].set(new_velocity_y)
