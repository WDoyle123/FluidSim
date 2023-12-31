import numpy as np
import math 
from calculations.kernels import spiky_smoothing_kernel_2d, spiky_smoothing_kernel_2d_derivative

class Particle:
    def __init__(self, x: float, y: float, mass=1.0, radius=1.0, x_velocity: float = 0., y_velocity: float = 0.):
        self.position = np.array([x, y])
        self.velocity = np.array([x_velocity, y_velocity])
        self.mass = mass
        self.radius = radius
        self.smoothing_radius = radius * 2
        self.density = 0
        self.pressure = np.array([0., 0.])

    def update_position(self, time_step, gravity=0):

        if gravity > 0: 
            self.velocity[1] += gravity * time_step

        self.position += self.velocity * time_step

    def apply_pressure_force(self, time_step):
        pressure_acceleration = self.pressure / (self.density + 1e-8)
        self.velocity += pressure_acceleration * time_step

    def boundary_collision(self, left, right, top, bottom, damping=1.0):

        # Handle collision with the left and right walls
        if self.position[0] - self.radius <= left:
            self.position[0] = left + self.radius
            self.velocity[0] = -self.velocity[0] * damping
        elif self.position[0] + self.radius >= right:
            self.position[0] = right - self.radius
            self.velocity[0] = -self.velocity[0] * damping

        # Handle collision with the top and bottom walls
        if self.position[1] - self.radius <= top:
            self.position[1] = top + self.radius
            self.velocity[1] = -self.velocity[1] * damping
        elif self.position[1] + self.radius >= bottom:
            self.position[1] = bottom - self.radius
            self.velocity[1] = -self.velocity[1] * damping
    
    def distance_to(self, other):

        delta = self.position - other.position
        distance = math.sqrt(delta[0] ** 2 + delta[1] ** 2)

        epsilon = 1e-8
        direction_vector = np.array([delta[0] / (distance + epsilon), delta[1] / (distance + epsilon)]) 
        return direction_vector, distance

    def calculate_density(self, distance):

        influence = spiky_smoothing_kernel_2d(distance, self.smoothing_radius)
        self.density += self.mass * influence

    def calculate_pressure_force(self, other, direction_vector, distance, target_density, pressure_coefficient):

        gradient_magnitude = spiky_smoothing_kernel_2d_derivative(distance, self.smoothing_radius)

        pressure_force_magnitude = (- other.mass * 
                                    (self.convert_density_to_pressure(target_density, pressure_coefficient)) /
                                    (other.density + 1e-8) *
                                    gradient_magnitude)

        pressure_force_vector = pressure_force_magnitude * direction_vector
        self.pressure += pressure_force_vector

    def convert_density_to_pressure(self, target_density, pressure_coefficient):
        density_error = self.density - target_density
        pressure = density_error * pressure_coefficient
        return pressure

    def collide_with(self, other, direction_vector, distance, positional_correction_factor=0.01, damping=1.0):

        if distance <= (self.smoothing_radius + other.smoothing_radius):
            relative_velocity = self.velocity - other.velocity
            velocity_along_normal = np.dot(relative_velocity, direction_vector)

            if velocity_along_normal > 0:
                return

            impulse = -(1 + damping) * velocity_along_normal / (1 / self.mass + 1 / other.mass)
            impulse_vector = impulse * direction_vector

            self.velocity += impulse_vector / self.mass
            other.velocity -= impulse_vector / other.mass
            
            '''
            # Positional correction (optional, based on your needs)
            correction_magnitude = positional_correction_factor * max(0, self.radius + other.radius - distance)
            correction_vector = correction_magnitude * direction_vector
            self.position += correction_vector / 2  # Half correction for each particle
            other.position -= correction_vector / 2
            '''
