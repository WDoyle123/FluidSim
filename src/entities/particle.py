import math

class Particle:
    def __init__(self, x: float, y: float, mass=1.0, radius=1.0, x_velocity: float = 0., y_velocity: float = 0.):
        self.x_position = x
        self.y_position = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.mass = mass
        self.radius = radius
    
    def update_position(self, time_step, gravity=9.81, floor=None, friction=0.9):
        self.x_position += self.x_velocity * time_step

        self.y_velocity += gravity * time_step
        self.y_position += self.y_velocity * time_step

        # Apply friction continuously when the ball is on the floor
        if floor is not None and self.y_position - self.radius <= floor:
            self.x_velocity *= friction

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
     
    def distance_to(self, other_particle):
        dx = self.x_position - other_particle.x_position
        dy = self.y_position - other_particle.y_position
        return math.sqrt(dx**2 + dy**2)

    def collide_with(self, other):
        dx = self.x_position - other.x_position
        dy = self.y_position - other.y_position
        distance = math.sqrt(dx**2 + dy**2)

        # Check for collision
        if distance < self.radius + other.radius:
            # Simple elastic collision logic for velocities
            self.x_velocity, other.x_velocity = other.x_velocity, self.x_velocity
            self.y_velocity, other.y_velocity = other.y_velocity, self.y_velocity

            # Calculate overlap
            overlap = 0.5 * (distance - self.radius - other.radius)

            # Displace current particle
            self.x_position -= overlap * (self.x_position - other.x_position) / (distance + 1e-6)
            self.y_position -= overlap * (self.y_position - other.y_position) / (distance + 1e-6)

            # Displace other particle
            other.x_position += overlap * (self.x_position - other.x_position) / (distance + 1e-6)
            other.y_position += overlap * (self.y_position - other.y_position) / (distance + 1e-6)

