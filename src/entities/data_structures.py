import pygame
import math

#########
# Notes #
#########

# From testing it seems that QuadTree is more efficent put has less accurate collision from the insert funtion

class QuadTree:
    def __init__(self, boundary, capacity):
        """
        Initialise the QuadTree.

        Args:
            boundary (pygame.Rect): The rectangular boundary of this node of the quadtree.
            capacity (int): The maximum number of particles that a node can hold before subdividing.
        """
        self.boundary = boundary  # Boundary of the quadtree node
        self.capacity = capacity  # Maximum number of particles in this node
        self.particles = []  # List to store particles
        self.divided = False  # Flag to check if this node has been subdivided

    def draw(self, screen):
        """
        Draw the boundaries of the QuadTree node and its subdivisions.

        Args:
            screen (pygame.Surface): The pygame surface on which to draw the QuadTree.
        """
        # Draw the boundary of the quadtree node
        pygame.draw.rect(screen, (255, 255, 255), self.boundary, 1)

        # Recursively draw the boundaries of the subdivided nodes
        if self.divided:
            self.north_west.draw(screen)
            self.north_east.draw(screen)
            self.south_west.draw(screen)
            self.south_east.draw(screen)

    def subdivide(self):
        """
        Subdivide the current QuadTree node into four smaller nodes.
        """
        # Calculate the dimensions for the subdivided nodes
        x, y, width, height = self.boundary
        half_width = width / 2
        half_height = height / 2

        # Create new boundaries for the subdivided nodes
        north_west = pygame.Rect(x, y, half_width, half_height)
        north_east = pygame.Rect(x + half_width, y, half_width, half_height)
        south_west = pygame.Rect(x, y + half_height, half_width, half_height)
        south_east = pygame.Rect(x + half_width, y + half_height, half_width, half_height)

        # Create new QuadTree nodes for each subdivided area
        self.north_west = QuadTree(north_west, self.capacity)
        self.north_east = QuadTree(north_east, self.capacity)
        self.south_west = QuadTree(south_west, self.capacity)
        self.south_east = QuadTree(south_east, self.capacity)
        self.divided = True  # Set the flag as the node is now subdivided

    def insert(self, particle):
        """
        Insert a particle into the QuadTree.

        Args:
            particle (Particle): The particle to be inserted.

        Returns:
            bool: True if the particle was successfully inserted, False otherwise.
        """
        # Check if the particle fits in the boundary after considering its radius
        if not self.boundary.inflate(particle.radius * 2, particle.radius * 2).collidepoint(particle.x_position, particle.y_position):
            return False  # Particle does not fit in this node

        # Add the particle to this node if capacity is not exceeded
        if len(self.particles) < self.capacity:
            self.particles.append(particle)
            return True

        # Subdivide the node if it's not already divided and attempt to insert the particle into the appropriate child node
        else:
            if not self.divided:
                self.subdivide()
            # Attempt to insert the particle into each of the subdivided nodes
            inserted = (self.north_west.insert(particle) or
                        self.north_east.insert(particle) or
                        self.south_west.insert(particle) or
                        self.south_east.insert(particle))
            return inserted  # Return whether the particle was successfully inserted

    def query(self, collision_box, found):
        """
        Find particles within a certain area.

        Args:
            collision_box (pygame.Rect): The area to search within.
            found (list): The list to which found particles will be added.
        """
        # Check if the search area intersects with this node's boundary
        if not self.boundary.colliderect(collision_box):
            return  # No intersection with the search area; stop searching this branch

        # Add particles in this node that are within the search area to the 'found' list
        else:
            for particle in self.particles:
                if collision_box.collidepoint(particle.x_position, particle.y_position):
                    found.append(particle)

            # Recursively search in subdivided nodes
            if self.divided:
                self.north_west.query(collision_box, found)
                self.north_east.query(collision_box, found)
                self.south_west.query(collision_box, found)
                self.south_east.query(collision_box, found)

class SpatialHashGrid:
    def __init__(self, boundary, total_number_of_particles, minimum_particle_radius):
        """
        Initialize the SpatialHashGrid.

        Args:
            boundary (tuple): A tuple of (x, y, width, height) defining the boundary of the grid.
            total_number_of_particles (int): Total number of particles expected in the grid. Used to calculate cell size.
            minimum_particle_radius (float): The radius of the smallest particle. Ensures cells are large enough to contain particles.
        """
        # Extract the boundary coordinates and dimensions
        self.x, self.y, self.width, self.height = boundary

        # Store total number of particles and the minimum particle radius
        self.total_number_of_particles = total_number_of_particles
        self.minimum_particle_radius = minimum_particle_radius

        # Calculate the average distance between particles and set cell dimensions
        average_distance = math.sqrt(self.width * self.height / (self.total_number_of_particles / 2))
        self.cell_dimension = max(average_distance, self.minimum_particle_radius * 2)

        # Initialize the dictionary to hold particles in each cell
        self.cells = {}

    def draw_grid(self, screen, color=(255, 255, 255)):
        """
        Draw the grid lines of the SpatialHashGrid on the screen.

        Args:
            screen (pygame.Surface): The pygame surface on which to draw the grid.
            color (tuple): The color to use for the grid lines, default is white.
        """
        # Draw vertical lines
        for x in range(int(self.x), int(self.x + self.width), int(self.cell_dimension)):
            pygame.draw.line(screen, color, (x, self.y), (x, self.y + self.height))

        # Draw horizontal lines
        for y in range(int(self.y), int(self.y + self.height), int(self.cell_dimension)):
            pygame.draw.line(screen, color, (self.x, y), (self.x + self.width, y))

    def calculate_cell_index(self, x, y):
        """
        Calculate the cell index for a given x, y position.

        Args:
            x (float): The x-coordinate of the position.
            y (float): The y-coordinate of the position.

        Returns:
            tuple: A tuple (cell_x, cell_y) representing the cell index.
        """
        return int(x / self.cell_dimension), int(y / self.cell_dimension)

    def insert_particle(self, particle):
        """
        Insert a particle into a SpatialHashGrid cell.

        Args:
            particle (Particle): The particle to be inserted.
        """
        # Calculate the cell index for the particle
        cell_index = self.calculate_cell_index(particle.x_position, particle.y_position)

        # Create a new cell if it doesn't exist
        if cell_index not in self.cells:
            self.cells[cell_index] = []

        # Add the particle to the cell
        self.cells[cell_index].append(particle)

    def get_neighbouring_cells(self, cell_index):
        """
        Get the indices of neighbouring cells around a given cell index.

        Args:
            cell_index (tuple): The cell index (cell_x, cell_y).

        Returns:
            list: A list of tuples representing the indices of neighbouring cells.
        """
        x, y = cell_index
        # Include the current cell and the eight surrounding cells
        return [(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2)]

    def get_potential_colliders(self, particle):
        """
        Get a list of potential collider particles for a given particle.

        Args:
            particle (Particle): The particle for which to find potential colliders.

        Returns:
            list: A list of particles that are potential colliders.
        """
        # Calculate the cell index for the particle
        cell_index = self.calculate_cell_index(particle.x_position, particle.y_position)
        potential_colliders = []

        # Check neighbouring cells for potential colliders
        for neighbouring_cell in self.get_neighbouring_cells(cell_index):
            if neighbouring_cell in self.cells:
                # Exclude the particle itself from potential colliders
                potential_colliders.extend([other_particle for other_particle in self.cells[neighbouring_cell] if other_particle != particle])

        return potential_colliders
