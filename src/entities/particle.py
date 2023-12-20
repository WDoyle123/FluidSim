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
            if self.x_velocity < 0.001:
                self.x_velocity = 0

    def box_collision(self, floor, ceiling, left, right, restitution=0.9):
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

    def distance_to(self, other_particle):
        dx = self.x_position - other_particle.x_position
        dy = self.y_position - other_particle.y_position
        return math.sqrt(dx**2 + dy**2)

    def collide_with(self, other, restitution=0.9, positional_correction_factor=0.01, damping=0.95):
        dx = self.x_position - other.x_position
        dy = self.y_position - other.y_position
        distance = math.sqrt(dx**2 + dy**2)

        # Check for collision
        if distance < self.radius + other.radius:
            # Normal vector
            nx = dx / (distance + 1e-6)
            ny = dy / (distance + 1e-6)

            # Relative velocity
            vx = self.x_velocity - other.x_velocity
            vy = self.y_velocity - other.y_velocity
    
            # Velocity component along the normal
            vel_normal = vx * nx + vy * ny

            # Check if particles are moving towards each other
            if vel_normal > 0:
                return

            # Conservation of momentum and inelastic collision
            impulse = (1 + restitution) * vel_normal / (1 / self.mass + 1 / other.mass)
            
            # Apply damping to reduce jitter
            self.x_velocity *= damping
            self.y_velocity *= damping
            other.x_velocity *= damping
            other.y_velocity *= damping

            # Positional correction to prevent overlap
            correction_amount = positional_correction_factor * max(0, self.radius + other.radius - distance)
            correction_vector_x = correction_amount * nx
            correction_vector_y = correction_amount * ny

            self.x_position += correction_vector_x
            self.y_position += correction_vector_y
            other.x_position -= correction_vector_x
            other.y_position -= correction_vector_y
