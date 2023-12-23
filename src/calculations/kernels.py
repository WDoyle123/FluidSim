import math
import numpy as np

def smoothing_kernel(radius, distance):

    if distance >= radius:
        return 0

    else:
        volume = math.pi * np.power(radius, 4) / 6
        return (radius - distance) ** 2 / volume

def smoothing_kernel_derivative(radius, distance):

    if (distance >= radius):
        return 0
    
    scale = 12 / (np.power(radius, 4) * math.pi)
    return (distance - radius) * scale

def simple_smoothing_kernel(radius, distance):
    return max(0, radius - distance)

