# FluidSim

## About 

**This project is currently under construction.**

FluidSim is a particle-based fluid simulation using Smoothed Particle Hydrodynamics (SPH) designed to demonstrate the concepts of fluid dynamics and collision detection in a 2D environment. The simulation utilises a SpatialHashGrid for optimised collision detection, reducing the computational complexity and enhancing the efficiency of the simulation. Additionally, FluidSim uses JAX to perform complex fluid calculation on the GPU.

## Features

- **Visualisations**:
  - Utilises pygame for dynamic, real-time visual representation of the fluid simulation.

- **JAX GPU Optimisation**:
  - Uses JAX code to run the complex fluid mechanics on GPUs for increased performance.

- **Real-time 2D Fluid Simulation**:
  - Demonstrates fluid movement and interactions in two dimensions.
  - Simulates various fluid behaviors and properties, capturing the essence of fluid dynamics.

- **Optimised Collision Detection**:
  - Implements a SpatialHashGrid for efficient and effective collision handling.
  - Significantly reduces computational overhead, particularly beneficial for large-scale simulations.

- **Advanced Fluid Dynamics**:
  - Employs the principles of Smoothed Particle Hydrodynamics (SPH) for realistic fluid motion.
  - Accurately calculates fluid density and pressure gradients for precise particle velocity adjustments.

- **Customisable Simulation Parameters**:
  - Allows users to modify simulation settings to observe different fluid behaviors.
  - Includes adjustable visual and physical parameters for a diverse range of simulation experiences.

## Installation

1. **Clone the Repository**:

```
git clone -b jax https://github.com/wdoyle123/FluidSim.git
```
2. **Install requirements.txt (optional virtual env)**:

```
cd FluidSim
pythom -m venv venv # optional
source venv/bin/activate # optional
pip install -r requirements.txt
```
Optionally for GPU computation on RTX 3070 (Driver Version: 545.29.06 CUDA Version: 12.3) (consult jax docs for specific hardware installation):

```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Demo

**Smooth Particle Hydrodynamics (SPH)**

In this fluid simulation, each particle's velocity vector is determined based on the local fluid density, following the SPH approach. The semi-transparent circles around each particle represent their area of influence, a key aspect of SPH.

Below are two demonstrations:

1. **Fluid Under Gravity**: 
   Observe how the fluid behaves under the influence of gravity, showcasing non-compressibility and fluid dynamics.
   
   ![smooth_particle_hydrodynamics.gif](figures/gravity_sim.gif)

2. **Fluid in Zero-Gravity**:
   Here, the fluid is simulated without gravity. Particles evenly distribute to maintain a uniform density, filling the available space.

   ![figure_1](figures/no_gravity_sim.gif)

## References

Müller, Matthias & Charypar, David & Gross, Markus. (2003). Particle-Based Fluid Simulation for Interactive Applications. Fluid Dynamics. 2003. 154-159. [Link to Paper](https://matthias-research.github.io/pages/publications/sca03.pdf)
