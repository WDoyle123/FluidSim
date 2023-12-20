import math

class Particle:
    def __init__(self, x: float, y: float, mass=1.0, radius=1.0, x_velocity: float = 0., y_velocity: float = 0.):
        self.x_position = x
        self.y_position = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.mass = mass
        self.radius = radius
        self.energy = 0
 
    def update_position(self, time_step, ceiling=None, gravity=9.81, friction=0.9):
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
        dx = self.x_position - other.x_position
        dy = self.y_position - other.y_position
        distance = math.sqrt(dx**2 + dy**2)

        return dx, dy, distance

    def collide_with(self, other, restitution=0.9, positional_correction_factor=0.01, damping=0.95):
        # Find distance between the two particles
        dx, dy, distance = self.distance_to(other)

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
