import math

def spiky_smoothing_kernel_2d(distance, smoothing_radius):
    """
    Calculate the 2D spiky smoothing kernel value for a given distance.

    This kernel function is commonly used in SPH (Smoothed Particle Hydrodynamics)
    simulations to model fluid dynamics. It is normalised to ensure that the
    integral over the area equals 1, which is crucial for accurately simulating
    fluid properties.

    :param distance: The distance between two particles.
    :type distance: float
    :param smoothing_radius: The smoothing radius (h) which defines the area of influence of the kernel.
    :type smoothing_radius: float
    :return: The value of the spiky smoothing kernel at the given distance.
    :rtype: float

    .. note::

        The kernel function is defined as 10 / (pi * h^5) * (h - r)^3 for 0 <= r <= h,
        where h is the smoothing radius and r is the distance. The kernel value is 0
        for distances greater than the smoothing radius.
    """
    if 0 <= distance <= smoothing_radius:
        normalisation_constant = 10 / (math.pi * smoothing_radius ** 5)
        spiky_kernel = (smoothing_radius - distance) ** 3

        return spiky_kernel * normalisation_constant
    else:
        return 0

def spiky_smoothing_kernel_2d_derivative(distance, smoothing_radius):
    """
    Calculate the derivative of the 2D spiky smoothing kernel with respect to distance.

    This derivative is used in SPH simulations for calculating gradients and
    forces. The derivative is calculated with respect to the distance between
    particles.

    :param distance: The distance between two particles.
    :type distance: float
    :param smoothing_radius: The smoothing radius (h) which defines the area of influence of the kernel.
    :type smoothing_radius: float
    :return: The derivative of the spiky smoothing kernel at the given distance.
    :rtype: float

    .. note::

        The derivative of the kernel function is defined as -30 / (pi * h^5) * (h - r)^2
        for 0 <= r <= h, where h is the smoothing radius and r is the distance. The
        derivative is 0 for distances greater than the smoothing radius.
    """
    if 0 <= distance <= smoothing_radius:
        normalisation_constant = -30 / (math.pi * smoothing_radius ** 5)
        spiky_kernel_derivative = (smoothing_radius - distance) ** 2

        return spiky_kernel_derivative * normalisation_constant
    else:
        return 0

import jax.numpy as jnp
from jax import jit, grad

@jit
def jax_spiky_smoothing_kernel_2d(distance, smoothing_radius):
    normalisation_constant = 10 / (jnp.pi * smoothing_radius ** 5)
    spiky_kernel = (smoothing_radius - distance) ** 3

    return jnp.where((0 <= distance) & (distance <= smoothing_radius), spiky_kernel * normalisation_constant, 0)

jax_spiky_smoothing_kernel_2d_derivative = grad(jax_spiky_smoothing_kernel_2d, argnums=0)
