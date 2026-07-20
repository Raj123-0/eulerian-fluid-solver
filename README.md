# Real-Time Eulerian Fluid Dynamics Solver

An interactive, real-time 2D fluid dynamics simulation written in pure Python. It solves the incompressible Navier-Stokes equations on a grid utilizing Jos Stam's Stable Fluids algorithm. 

To maintain 60 FPS while running computationally heavy physics calculations, the backend is heavily optimized using NumPy vectorization and Numba Just-In-Time (JIT) compilation to bypass the Python Global Interpreter Lock (GIL) and enable multi-core parallelization.

## Features
- **Incompressible Flow:** Accurate calculation of the Poisson equation to maintain zero divergence in the velocity field.
- **JIT Compilation:** `@njit(parallel=True, fastmath=True)` decorators allow massive mathematical loops to execute at C-speeds.
- **Interactive Visualization:** Real-time fluid rendering built with Pygame's `surfarray` for direct array-to-GPU memory processing.
- **Dynamic Controls:** Adjust viscosity and diffusion matrices on the fly using keyboard inputs.

## Installation and Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/Raj123-0/eulerian-fluid-solver.git
   cd eulerian-fluid-solver
   ```
2. Install the required dependencies:
  ```bash
    pip install -r requirements.txt 
```
3. Run the simulation:
  ```bash
  python main.py
```
Controls
Mouse Left Click & Drag: Inject dye and velocity into the grid.

UP / DOWN Arrows: Increase or decrease fluid Viscosity.

LEFT / RIGHT Arrows: Increase or decrease fluid Diffusi
