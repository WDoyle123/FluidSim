import jax.numpy as jnp
from jax import jit


@jit
def distance_to(position1, position2):
    delta = position1 - position2
    distance = jnp.sqrt(jnp.sum(delta ** 2))

    epsilon = 1e-8
    direction_vector = delta / (distance + epsilon)
    return direction_vector, distance
