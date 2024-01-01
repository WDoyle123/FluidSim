import numpy as np
import math 
from calculations.kernels import spiky_smoothing_kernel_2d, spiky_smoothing_kernel_2d_derivative

class Particle:
    """
    A class to represent a particle in a fluid simulation.

    This class defines the properties and initial state of a particle in the simulation.
    It includes physical attributes like position, velocity, mass, and radius, along with
    properties used for fluid dynamics calculations such as smoothing radius, density,
    and pressure.
    """

    def __init__(self, x: float, y: float, mass=1.0, radius=1.0, x_velocity: float = 0., y_velocity: float = 0.):
        """
        Initialises a Particle object with specified position, mass, radius, and velocity.

        :param x: The x-coordinate of the particle's initial position.
        :type x: float
        :param y: The y-coordinate of the particle's initial position.
        :type y: float
        :param mass: The mass of the particle, defaults to 1.0.
        :type mass: float, optional
        :param radius: The radius of the particle, defaults to 1.0.
        :type radius: float, optional
        :param x_velocity: The initial x-velocity of the particle, defaults to 0.
        :type x_velocity: float, optional
        :param y_velocity: The initial y-velocity of the particle, defaults to 0.
        :type y_velocity: float, optional
        """
        self.position = np.array([x, y])
        self.velocity = np.array([x_velocity, y_velocity])
        self.mass = mass
        self.radius = radius
        self.smoothing_radius = radius * 4
        self.density = 0
        self.pressure = np.array([0., 0.])

    def update_position(self, time_step, gravity=0):
        """
        Updates the particle's position based on its velocity and the effect of gravity.

        This method calculates the new position of the particle by considering its current velocity
        and the influence of gravity (if any). The gravity effect is only applied in the y-direction.
        The particle's position is updated based on the elapsed time step.

        :param time_step: The time step for which the particle's position is updated.
        :type time_step: float
        :param gravity: The gravitational acceleration affecting the particle's y-velocity, defaults to 0.
                        Positive values of gravity will result in downward acceleration.
        :type gravity: float, optional
        """
        if gravity > 0: 
            self.velocity[1] += gravity * time_step

        self.position += self.velocity * time_step

    def apply_pressure_force(self, time_step):
        """
        Updates the particle's velocity based on the pressure force.

        This method calculates the acceleration due to pressure by dividing the pressure
        by the particle's density (with a small value added to avoid division by zero).
        It then updates the particle's velocity by this pressure-induced acceleration
        multiplied by the time step.

        The method ensures that changes in pressure directly influence the particle's motion,
        adhering to fluid dynamics principles.

        :param time_step: The time step for which the particle's velocity is updated.
        :type time_step: float
        """
        pressure_acceleration = self.pressure / (self.density + 1e-8)
        self.velocity += pressure_acceleration * time_step

    def boundary_collision(self, left, right, top, bottom, damping=1.0):
        """
        Handles collisions between the particle and the boundary walls.

        This method checks if the particle has collided with any of the four walls defined by
        the left, right, top, and bottom boundaries. If a collision is detected, the particle's
        position is adjusted to remain within the boundaries, and its velocity is reflected and
        dampened according to the damping factor.

        Collisions are considered elastic, with the option to introduce energy loss through damping.

        :param left: The x-coordinate of the left boundary.
        :type left: float
        :param right: The x-coordinate of the right boundary.
        :type right: float
        :param top: The y-coordinate of the top boundary.
        :type top: float
        :param bottom: The y-coordinate of the bottom boundary.
        :type bottom: float
        :param damping: The damping factor applied to the velocity after collision, defaults to 1.0 (no energy loss).
        :type damping: float, optional
        """
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
        """
        Calculates the direction vector and distance to another particle.

        This method computes the Euclidean distance between the current particle and another
        particle, and also determines the direction vector pointing from this particle to the other.
        The direction vector is normalized to have a unit length.

        To avoid division by zero in the direction vector calculation, a small epsilon value is added
        to the distance.

        :param other: The other particle to which the distance and direction are calculated.
        :type other: Particle
        :return: A tuple containing the normalised direction vector and the distance to the other particle.
        :rtype: (numpy.ndarray, float)
        """
        # Position difference
        delta = self.position - other.position
        distance = math.sqrt(delta[0] ** 2 + delta[1] ** 2)

        # Normalised unit vector for direction
        epsilon = 1e-8
        direction_vector = np.array([delta[0] / (distance + epsilon), delta[1] / (distance + epsilon)]) 
        return direction_vector, distance

    def calculate_density(self, distance):
        """
        Calculates and updates the particle's density based on its distance from another particle.

        This method uses a smoothing kernel function, specifically the spiky smoothing kernel in 2D, 
        to compute the influence of another particle at a given distance. The particle's density is 
        then updated by adding the product of its mass and this influence.

        The smoothing kernel function is key in simulating the fluid-like behavior of the particles,
        as it determines how the presence of nearby particles affects the density of this particle.

        :param distance: The distance between this particle and another particle.
        :type distance: float
        """
        influence = spiky_smoothing_kernel_2d(distance, self.smoothing_radius)
        self.density += self.mass * influence

    def calculate_pressure_force(self, other, direction_vector, distance, target_density, pressure_coefficient):
        """
        Calculates and updates the pressure force exerted on this particle by another particle.

        This method uses a derivative of the spiky smoothing kernel function to calculate the gradient magnitude 
        of the pressure force based on the distance to another particle. It then computes the pressure force 
        vector exerted by the other particle on this one, taking into account the mass of the other particle, 
        the converted pressure of this particle, and the density of the other particle.

        The pressure force contributes to the particle's overall force, influencing its motion and interaction 
        with other particles in the fluid simulation.

        :param other: The other particle exerting the pressure force.
        :type other: Particle
        :param direction_vector: The normalised direction vector from this particle to the other particle.
        :type direction_vector: numpy.ndarray
        :param distance: The distance between this particle and the other particle.
        :type distance: float
        :param target_density: The target density for the fluid simulation.
        :type target_density: float
        :param pressure_coefficient: The coefficient used in pressure conversion calculations.
        :type pressure_coefficient: float
        """
        gradient_magnitude = spiky_smoothing_kernel_2d_derivative(distance, self.smoothing_radius)

        pressure_force_magnitude = (- other.mass * 
                                    (self.convert_density_to_pressure(target_density, pressure_coefficient)) /
                                    (other.density + 1e-8) *
                                    gradient_magnitude)

        pressure_force_vector = pressure_force_magnitude * direction_vector
        self.pressure += pressure_force_vector

    def convert_density_to_pressure(self, target_density, pressure_coefficient):
        """
        Converts the particle's density to pressure.

        This method calculates the pressure based on the particle's current density, 
        a target density, and a pressure coefficient. The pressure is derived from 
        the difference between the particle's density and the target density, 
        multiplied by the pressure coefficient.

        This conversion is crucial in fluid dynamics simulations, as it helps in 
        determining the pressure forces acting on the particles based on their 
        densities and the overall fluid properties.

        :param target_density: The target density for the fluid simulation.
        :type target_density: float
        :param pressure_coefficient: The coefficient used in converting density to pressure.
        :type pressure_coefficient: float
        :return: The calculated pressure for the particle.
        :rtype: float
        """
        density_error = self.density - target_density
        pressure = density_error * pressure_coefficient
        return pressure

    def collide_with(self, other, direction_vector, distance, positional_correction_factor=0.001, damping=1.0):
        """
        Handles collision response between this particle and another particle.

        This method is called when two particles are close enough to collide (i.e., when the distance 
        between them is less than the sum of their smoothing radii). It computes the impulse caused by 
        the collision and adjusts the velocities of both particles accordingly.

        The collision is treated as an inelastic collision where the damping factor can reduce the relative 
        velocity along the collision normal. The positional correction factor is used to avoid the 'sinking' 
        effect often seen in simulations.

        :param other: The other particle involved in the collision.
        :type other: Particle
        :param direction_vector: The normalized direction vector from this particle to the other particle.
        :type direction_vector: numpy.ndarray
        :param distance: The distance between this particle and the other particle.
        :type distance: float
        :param positional_correction_factor: Factor used for position correction, defaults to 0.001.
        :type positional_correction_factor: float, optional
        :param damping: The damping factor applied to the collision, defaults to 1.0 (elastic collision).
        :type damping: float, optional
        """
        if distance <= (self.smoothing_radius + other.smoothing_radius):
            relative_velocity = self.velocity - other.velocity
            velocity_along_normal = np.dot(relative_velocity, direction_vector)

            if velocity_along_normal > 0:
                return

            impulse = -(1 + damping) * velocity_along_normal / (1 / self.mass + 1 / other.mass)
            impulse_vector = impulse * direction_vector

            self.velocity += impulse_vector / self.mass
            other.velocity -= impulse_vector / other.mass
            
            correction_magnitude = positional_correction_factor * max(0, self.smoothing_radius + other.smoothing_radius - distance)
            correction_vector = correction_magnitude * direction_vector
            self.position += correction_vector / 2  # Half correction for each particle
            other.position -= correction_vector / 2
            
