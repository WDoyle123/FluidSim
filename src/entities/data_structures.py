import math

class SpatialHashGrid:
    """
    A spatial hash grid for efficient management of particles in a simulation.

    This class provides a way to divide the simulation space into a grid of cells, 
    allowing for efficient querying of particles based on their spatial location. 
    It's particularly useful for collision detection and finding nearby particles 
    without having to check every particle pair.

    Significantly reduces computational load.
    """
    def __init__(self, boundary, particle_count, particle_radius):
        """
        Initialises the SpatialHashGrid with the given boundary, particle count, and particle radius.

        :param boundary: The boundary of the simulation space.
        :type boundary: pygame.Rect
        :param particle_count: The number of particles in the simulation.
        :type particle_count: int
        :param particle_radius: The radius of the particles.
        :type particle_radius: float
        """
        self.boundary = boundary
        self.particle_count = particle_count
        self.particle_radius = particle_radius
        self.average_distance = math.sqrt(self.boundary.right * self.boundary.bottom / (self.particle_count / 2))
        self.cell_dimension = max(self.average_distance, self.particle_radius)
        self.cells = {}

    def draw_grid(self, screen, color=(255, 255, 255)):
        """
        Draws the grid on a Pygame screen.

        This method visualises the grid lines of the spatial hash grid on the provided screen.
        It can be useful for debugging or visualising the spatial partitioning.

        :param screen: The Pygame screen on which to draw the grid.
        :type screen: pygame.Surface
        :param color: The color of the grid lines, defaults to white.
        :type color: tuple, optional
        """
        # Draw vertical lines
        for x in range(int(self.boundary.left), int(self.boundary.right), int(self.cell_dimension)):
            pygame.draw.line(screen, color, (x, self.boundary.top), (x, self.boundary.bottom))

        # Draw horizontal lines
        for y in range(int(self.boundary.top), int(self.boundary.bottom), int(self.cell_dimension)):
            pygame.draw.line(screen, color, (self.boundary.left, y), (self.boundary.right, y))

    def calculate_cell_index(self, position):
        """
        Calculates the cell index in the grid for a given position.

        :param position: The position for which to calculate the cell index.
        :type position: tuple
        :return: The cell index corresponding to the position.
        :rtype: tuple
        """
        x, y = position
        return int(x / self.cell_dimension), int(y / self.cell_dimension)

    def insert_particle(self, particle):
        """
        Inserts a particle into the appropriate cell in the grid.

        :param particle: The particle to be inserted.
        :type particle: Particle
        """
        cell_index = self.calculate_cell_index(particle.position)

        if cell_index not in self.cells:
            self.cells[cell_index] = []

        self.cells[cell_index].append(particle)

    def get_neighbouring_cells(self, cell_index):
        """
        Retrieves the indices of neighbouring cells to a given cell.

        :param cell_index: The index of the cell for which to find neighbours.
        :type cell_index: tuple
        :return: A list of indices of neighbouring cells.
        :rtype: list of tuples
        """
        x, y = cell_index
        return [(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2)]

    def nearby_to(self, particle):
        """
        Finds and returns particles that are nearby to a given particle.

        This method identifies the cell of the given particle and then retrieves particles
        from neighbouring cells, effectively finding particles that are spatially close.

        :param particle: The particle for which to find nearby particles.
        :type particle: Particle
        :return: A list of particles that are nearby to the given particle.
        :rtype: list of Particles
        """
        cell_index = self.calculate_cell_index(particle.position)
        nearby_particles = []

        for neighbouring_cell in self.get_neighbouring_cells(cell_index):
            if neighbouring_cell in self.cells:
                nearby_particles.extend([other_particle for other_particle in self.cells[neighbouring_cell] if other_particle != particle])

        return nearby_particles
