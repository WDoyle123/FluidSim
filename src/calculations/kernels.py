import math
import numpy as np

def smoothing_kernel(radius, distance):

    if distance >= radius:
        return 0

    else:
        volume = math.pi * np.power(radius, 4) / 6
        return (radius - distance) ** 2 / volume

def simple_smoothing_kernel(radius, distance):
    return max(0, radius - distance)
