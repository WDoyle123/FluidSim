import math

class Particle:
    """
    Represents a physical particle with properties like position, velocity, mass, and radius.

    Attributes:
        x_position (float): The x-coordinate of the particle.
        y_position (float): The y-coordinate of the particle.
        x_velocity (float): The velocity of the particle along the x-axis.
        y_velocity (float): The velocity of the particle along the y-axis.
        mass (float): The mass of the particle.
        radius (float): The radius of the particle.
        energy (float): The total mechanical energy of the particle.

    """
    def __init__(self, x: float, y: float, mass=1.0, radius=1.0, x_velocity: float = 0., y_velocity: float = 0.):
        self.x_position = x
        self.y_position = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.mass = mass
        self.radius = radius
        self.energy = 0
 
    def update_position(self, time_step, ceiling=None, gravity=9.81, friction=0.9):
        """
        Updates the position of the particle based on its velocity, applying gravitational and frictional forces.

        Args:
            time_step (float): The time step for the update.
            ceiling (float, optional): The y-coordinate representing the ceiling. If the particle hits the ceiling, it will bounce back. Defaults to None.
            gravity (float, optional): The gravitational acceleration. Defaults to 9.81.
            friction (float, optional): The friction coefficient to apply when the particle is in contact with the ceiling. Defaults to 0.9.
        """
        self.x_position += self.x_velocity * time_step

        self.y_velocity += gravity * time_step
        self.y_position += self.y_velocity * time_step
        self.update_energy()

        # Apply friction continuously when the ball is on the floor
        if ceiling is not None and self.y_position + self.radius >= ceiling:
            self.x_velocity *= friction
            if self.x_velocity < 1e-12:
                self.x_velocity = 0

    def boundary_collision(self, floor, ceiling, left, right, restitution=0.9):
        """
        Handles the particle's collision with boundary walls (floor, ceiling, left, right).

        Args:
            floor (float): The y-coordinate of the floor.
            ceiling (float): The y-coordinate of the ceiling.
            left (float): The x-coordinate of the left wall.
            right (float): The x-coordinate of the right wall.
            restitution (float, optional): The restitution coefficient for the collision. Defaults to 0.9.
        """
        # Top and Bottom Walls
        if self.y_position - self.radius <= floor:
            self.y_position = floor + self.radius
            self.y_velocity = -self.y_velocity * restitution

        elif self.y_position + self.radius >= ceiling:
            self.y_position = ceiling - self.radius
            self.y_velocity = -self.y_velocity * restitution

        # Left and Right Walls
        if self.x_position - self.radius <= left:
            self.x_position = left + self.radius
            self.x_velocity = -self.x_velocity * restitution
        elif self.x_position + self.radius >= right:
            self.x_position = right - self.radius
            self.x_velocity = -self.x_velocity * restitution

    def update_energy(self, gravity=9.81, ground_y=600):
        """
        Updates the total mechanical energy of the particle based on its current position and velocity.

        Args:
            gravity (float, optional): The gravitational acceleration. Defaults to 9.81.
            ground_y (float, optional): The y-coordinate of the ground in the simulation environment. Defaults to 600.
        """
        # Reset energy
        self.energy = 0

        # Calculate kinetic energy
        velocity_squared = self.x_velocity**2 + self.y_velocity**2
        kinetic_energy = 0.5 * self.mass * velocity_squared
        self.energy += kinetic_energy

        # Calculate potential energy
        # ground_y is the y-coordinate of the ground in the Pygame window
        potential_energy = self.mass * gravity * (ground_y - self.y_position)
        self.energy += potential_energy

    def distance_to(self, other):
        """
        Calculates the distance between this particle and another particle.

        Args:
            other (Particle): The other particle to calculate the distance to.

        Returns:
            tuple of (float, float, float): The delta x, delta y, and the distance between the two particles.
        """
        dx = self.x_position - other.x_position
        dy = self.y_position - other.y_position
        distance = math.sqrt(dx**2 + dy**2)

        return dx, dy, distance

    def collide_with(self, other, dx, dy, distance, restitution=0.9, positional_correction_factor=0.01, damping=0.95):
        """
        Handles the collision between this particle and another particle.

        Args:
            other (Particle): The other particle to check collision with.
            restitution (float, optional): The restitution coefficient for the collision. Defaults to 0.9.
            positional_correction_factor (float, optional): Factor used to correct the position of colliding particles. Defaults to 0.01.
            damping (float, optional): Damping factor to apply to the velocity after collision. Defaults to 0.95.
        """
        # Check to see if they collide
        if distance <= (self.radius + other.radius):

            # Get normal vectors
            normal_x = dx / (distance + 1e-6)
            normal_y = dy / (distance + 1e-6)

            # Relative velocities
            x_relative_velocity = self.x_velocity - other.x_velocity
            y_relative_velocity = self.y_velocity - other.y_velocity

            # Velocity along the normal
            velocity_along_normal = (x_relative_velocity * normal_x) + (y_relative_velocity * normal_y)

            # Return if particles are not going to collide
            if velocity_along_normal > 0:
                return

            # Calculate the impulse
            impulse = ((1 + restitution) * velocity_along_normal) / (1 / self.mass + 1 / other.mass)

            # Apply impulse
            x_impulse = impulse * normal_x
            y_impulse = impulse * normal_y

            self.x_velocity -= x_impulse / self.mass
            self.y_velocity -= y_impulse / self.mass

            other.x_velocity += x_impulse / other.mass
            other.y_velocity += y_impulse / other.mass

            self.x_velocity *= damping
            self.y_velocity *= damping
    
            other.x_velocity *= damping
            other.y_velocity *= damping
