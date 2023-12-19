import pygame

class Quadtree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary
        self.capacity = capacity
        self.particles = []
        self.divided = False

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.boundary, 1)
        if self.divided:
            self.north_west.draw(screen)
            self.north_east.draw(screen)
            self.south_west.draw(screen)
            self.south_east.draw(screen)

    def subdivide(self):
        x, y, width, height = self.boundary
        half_width = width / 2
        half_height = height / 2

        north_west = pygame.Rect(x, y, half_width, half_height)
        north_east = pygame.Rect(x + half_width, y, half_width, half_height)
        south_west = pygame.Rect(x, y + half_height, half_width, half_height)
        south_east = pygame.Rect(x + half_width, y + half_height, half_width, half_height)

        self.north_west = Quadtree(north_west, self.capacity)
        self.north_east = Quadtree(north_east, self.capacity)
        self.south_west = Quadtree(south_west, self.capacity)
        self.south_east = Quadtree(south_east, self.capacity)
        self.divided = True

    def insert(self, particle):
        if not self.boundary.collidepoint(particle.x_position, particle.y_position):
            return False

        if len(self.particles) < self.capacity:
            self.particles.append(particle)
            return True

        else:
            if not self.divided:
                self.subdivide()

            if self.north_west.insert(particle): return True
            if self.north_east.insert(particle): return True
            if self.south_west.insert(particle): return True
            if self.south_east.insert(particle): return True

    def query(self, range, found):
        if not self.boundary.colliderect(range):
            return
        else:
            for particle in self.particles:
                if range.collidepoint(particle.x_position, particle.y_position):
                    found.append(particle)

            if self.divided:
                self.north_west.query(range, found)
                self.north_east.query(range, found)
                self.south_west.query(range, found)
                self.south_east.query(range, found)
