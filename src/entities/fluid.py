import random
import pygame
from entities.particle import Particle
from entities.data_structures import SpatialHashGrid, QuadTree

class Fluid:
    """
    Represents a fluid composed of many particles.

    Attributes:
        particles (list of Particle): The particles that make up the fluid.
        width (int): Width of the area in which the fluid is contained.
        height (int): Height of the area in which the fluid is contained.
    """
    def __init__(self, boundary, num_particles, restitution = 1.0, gravity = 10, friction = 1.0, data_structure='SpatialHashGrid'):
        """
        Initialise the fluid with a set number of particles.

        Args:
            width (int): Width of the area in which the fluid is contained.
            height (int): Height of the area in which the fluid is contained.
            num_particles (int): Number of particles in the fluid.
        """
        self.boundary = boundary
        _, _, self.width, self.height = self.boundary
        self.num_particles = num_particles
        self.particles = [self._create_particle() for _ in range(self.num_particles)]
        self.restitution = restitution
        self.gravity = gravity
        self.friction = friction
        if data_structure == 'SpatialHashGrid':
            self.SHG = SpatialHashGrid(self.boundary, num_particles, self.particles[0].smoothing_radius)
        elif data_structure == 'QuadTree':
            self.QT = QuadTree(self.boundary, capacity=16)
        else:
            print('Data Structure Not Recognised!')

    def _create_particle(self):
        """
        Create a single particle with random properties.

        Returns:
            Particle: A new particle instance.
        """
        x = random.uniform(self.width/2, self.width)
        y = random.uniform(self.height/2, self.height)
        mass = 1.0
        radius = 5.0
        x_velocity = random.uniform(-0, 0)
        y_velocity = random.uniform(-0, 0)
        return Particle(x, y, mass, radius, x_velocity, y_velocity)

    def update_SHG(self, time_step):
        """
        Update the state of the fluid for the next frame.

        Args:
            time_step (float): The time step for the update.
        """
        densities = self.calculate_quadrant_densities()
        print(f'\r{densities}', end='')
        self.SHG = SpatialHashGrid(self.boundary, self.num_particles, self.particles[0].radius)
        for particle in self.particles:
            particle.update_position(time_step, gravity=self.gravity, friction=self.friction)
            particle.boundary_collision(0, self.height, 0, self.width, restitution=self.restitution)

            # Insert particles into SpatialHashGrid and handle collisions
            self.SHG.insert_particle(particle)
            nearby_particles = self.SHG.get_potential_colliders(particle)
            for other in nearby_particles:
                dx, dy, distance = particle.distance_to(other)
                particle.calculate_density_derivative(other, dx, dy, distance)
                particle.apply_pressure_force(time_step)
                if distance < particle.radius + other.radius:
                   # Uncomment for Collisions!
                   #particle.collide_with(other, dx, dy, distance)
                   pass
              
    def update_QT(self, time_step):
        """
        Update the state of the fluid for the next frame.

        Args:
            time_step (float): The time step for the update.
        """
        densities = self.calculate_quadrant_densities()
        print(f'\r{densities}', end='')
        self.QT = QuadTree(self.boundary, capacity=16)
        for particle in self.particles:
            particle.update_position(time_step, gravity=self.gravity, friction=self.friction)
            particle.boundary_collision(0, self.height, 0, self.width, restitution=self.restitution)

            # Insert particles into SpatialHashGrid and handle collisions
            self.QT.insert_particle(particle)

            collision_box = pygame.Rect(particle.x_position - particle.smoothing_radius,
                                        particle.y_position - particle.smoothing_radius,
                                        particle.smoothing_radius * 2, particle.smoothing_radius * 2)

            nearby_particles = []
            self.QT.query(collision_box, nearby_particles)
            for other in nearby_particles:
                dx, dy, distance = particle.distance_to(other)
                particle.calculate_density_derivative(other, dx, dy, distance)
                particle.apply_pressure_force(time_step)
                if distance < particle.radius + other.radius:
                   # Uncomment for Collisions!
                   # particle.collide_with(other, dx, dy, distance)
                   pass
               

    def draw(self, screen):
        """
        Draw the fluid particles on the screen.
    
        Args:
            screen (pygame.Surface): The pygame surface to draw the fluid on.
        """

        # Then draw the smaller blue circles on top without transparency
        for particle in self.particles:
            pygame.draw.circle(screen, (50, 50, 255), 
                               (int(particle.x_position), int(particle.y_position)), 
                               particle.radius)

            # Calculate the end point of the gradient arrow
            grad_length = 10  # Adjust the length of the gradient arrow
            end_x = int(particle.x_position + particle.x_gradient * grad_length)
            end_y = int(particle.y_position + particle.y_gradient * grad_length)

            # Draw the gradient arrow
            pygame.draw.line(screen, (255, 0, 0), (int(particle.x_position), int(particle.y_position)), (end_x, end_y), 2)
            pygame.draw.polygon(screen, (255, 0, 0), 
                                [(end_x, end_y),
                                 (end_x - 3, end_y + 6), 
                                 (end_x + 3, end_y + 6)])

        red_alpha = 64  # 25% transparency

        for particle in self.particles:
            # Create a temporary surface that's large enough to hold the entire circle
            temp_surface = pygame.Surface((particle.smoothing_radius * 2, particle.smoothing_radius * 2), pygame.SRCALPHA)
            temp_surface.fill((0, 0, 0, 0))  # Fill with a fully transparent color
    
            # Draw the semi-transparent red circle onto the temporary surface
            # Note that the center of the circle is at (smoothing_radius, smoothing_radius)
            pygame.draw.circle(temp_surface, (50, 50, 255, red_alpha), 
                           (particle.smoothing_radius, particle.smoothing_radius), 
                           particle.smoothing_radius)

            # Blit this surface onto the main screen surface
            # Note that we offset the blit position by smoothing_radius to properly center it
            screen.blit(temp_surface, (int(particle.x_position - particle.smoothing_radius), 
                                       int(particle.y_position - particle.smoothing_radius)))

    def calculate_quadrant_densities(self):
        """
        Calculate the density of particles in each quadrant of the container.

        Returns:
            dict: A dictionary containing the density of each quadrant.
        """
        densities = {'top_left': 0, 'top_right': 0, 'bottom_left': 0, 'bottom_right': 0}
        quadrant_width = self.width / 2
        quadrant_height = self.height / 2

        for particle in self.particles:
            if particle.x_position < quadrant_width and particle.y_position < quadrant_height:
                densities['top_left'] += particle.density
            elif particle.x_position >= quadrant_width and particle.y_position < quadrant_height:
                densities['top_right'] += particle.density
            elif particle.x_position < quadrant_width and particle.y_position >= quadrant_height:
                densities['bottom_left'] += particle.density
            else:  # particle.x_position >= quadrant_width and particle.y_position >= quadrant_height
                densities['bottom_right'] += particle.density

        return densities




