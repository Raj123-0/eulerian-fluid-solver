import numpy as np
import time
from numba import njit, prange

def profile_func(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            wrapper.total_time += (end_time - start_time)
            wrapper.calls += 1
            return result
        wrapper.total_time = 0.0
        wrapper.calls = 0
        return wrapper
    return decorator

# Enforces boundary conditions so fluid doesn't leak off-screen.
@njit(fastmath=True)
def set_bnd(b, x, Nx, Ny):
    for i in range(1, Nx + 1):
        x[i, 0] = -x[i, 1] if b == 2 else x[i, 1]
        x[i, Ny + 1] = -x[i, Ny] if b == 2 else x[i, Ny]
        
    for j in range(1, Ny + 1):
        x[0, j] = -x[1, j] if b == 1 else x[1, j]
        x[Nx + 1, j] = -x[Nx, j] if b == 1 else x[Nx, j]
        
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, Ny + 1] = 0.5 * (x[1, Ny + 1] + x[0, Ny])
    x[Nx + 1, 0] = 0.5 * (x[Nx, 0] + x[Nx + 1, 1])
    x[Nx + 1, Ny + 1] = 0.5 * (x[Nx, Ny + 1] + x[Nx + 1, Ny])

# Solves linear systems used for diffusion and projection.
@njit(parallel=True, fastmath=True)
def lin_solve(b, x, x0, a, c, iter_count, Nx, Ny):
    c_recip = 1.0 / c
    temp = np.empty_like(x)
    for _ in range(iter_count):
        for i in prange(1, Nx + 1):
            for j in range(1, Ny + 1):
                temp[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) * c_recip
        
        for i in prange(1, Nx + 1):
            for j in range(1, Ny + 1):
                x[i, j] = temp[i, j]
                
        set_bnd(b, x, Nx, Ny)

@njit(fastmath=True)
def diffuse_numba(b, x, x0, diff, dt, iter_count, Nx, Ny):
    a = dt * diff * max(Nx, Ny) * max(Nx, Ny)
    if a == 0:
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                x[i, j] = x0[i, j]
        set_bnd(b, x, Nx, Ny)
        return
    lin_solve(b, x, x0, a, 1 + 4 * a, iter_count, Nx, Ny)

@profile_func("diffuse")
def diffuse(b, x, x0, diff, dt, iter_count, Nx, Ny):
    diffuse_numba(b, x, x0, diff, dt, iter_count, Nx, Ny)

# Ensures the fluid is mass-conserving (incompressible).
@njit(parallel=True, fastmath=True)
def project_numba(u, v, p, div, iter_count, Nx, Ny):
    scale = float(max(Nx, Ny))
    
    for i in prange(1, Nx + 1):
        for j in range(1, Ny + 1):
            div[i, j] = -0.5 * (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1]) / scale
            p[i, j] = 0.0
            
    set_bnd(0, div, Nx, Ny)
    set_bnd(0, p, Nx, Ny)
    
    lin_solve(0, p, div, 1.0, 4.0, iter_count, Nx, Ny)
    
    for i in prange(1, Nx + 1):
        for j in range(1, Ny + 1):
            u[i, j] -= 0.5 * scale * (p[i + 1, j] - p[i - 1, j])
            v[i, j] -= 0.5 * scale * (p[i, j + 1] - p[i, j - 1])
            
    set_bnd(1, u, Nx, Ny)
    set_bnd(2, v, Nx, Ny)

def project(u, v, p, div, iter_count, Nx, Ny):
    project_numba(u, v, p, div, iter_count, Nx, Ny)

# Moves quantities (like density or velocity) along the fluid's flow.
@njit(parallel=True, fastmath=True)
def advect_numba(b, d, d0, u, v, dt, Nx, Ny):
    dt0 = dt * max(Nx, Ny)
    
    for i in prange(1, Nx + 1):
        for j in range(1, Ny + 1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            
            if x < 0.5: x = 0.5
            if x > Nx + 0.5: x = Nx + 0.5
            if y < 0.5: y = 0.5
            if y > Ny + 0.5: y = Ny + 0.5
            
            i0 = int(x)
            j0 = int(y)
            i1 = i0 + 1
            j1 = j0 + 1
            
            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1
            
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + \
                      s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
                      
    set_bnd(b, d, Nx, Ny)

@profile_func("advect")
def advect(b, d, d0, u, v, dt, Nx, Ny):
    advect_numba(b, d, d0, u, v, dt, Nx, Ny)
