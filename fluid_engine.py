import numpy as np
from math_kernels import diffuse, project, advect

class Fluid:
    def __init__(self, Nx, Ny, iter_count=16, diff=0.0, visc=0.0):
        self.Nx = Nx
        self.Ny = Ny
        self.iter = iter_count
        self.diff = diff
        self.visc = visc

        # CPU Arrays (NumPy)
        self.s = np.zeros((3, Nx + 2, Ny + 2), dtype=np.float32)
        self.density = np.zeros((3, Nx + 2, Ny + 2), dtype=np.float32)

        self.Vx = np.zeros((Nx + 2, Ny + 2), dtype=np.float32)
        self.Vy = np.zeros((Nx + 2, Ny + 2), dtype=np.float32)
        self.Vx0 = np.zeros((Nx + 2, Ny + 2), dtype=np.float32)
        self.Vy0 = np.zeros((Nx + 2, Ny + 2), dtype=np.float32)
        
    def resize(self, new_Nx, new_Ny):
        if self.Nx == new_Nx and self.Ny == new_Ny:
            return
            
        new_s = np.zeros((3, new_Nx + 2, new_Ny + 2), dtype=np.float32)
        new_density = np.zeros((3, new_Nx + 2, new_Ny + 2), dtype=np.float32)

        new_Vx = np.zeros((new_Nx + 2, new_Ny + 2), dtype=np.float32)
        new_Vy = np.zeros((new_Nx + 2, new_Ny + 2), dtype=np.float32)
        new_Vx0 = np.zeros((new_Nx + 2, new_Ny + 2), dtype=np.float32)
        new_Vy0 = np.zeros((new_Nx + 2, new_Ny + 2), dtype=np.float32)

        # Copy existing data where it fits
        copy_Nx = min(self.Nx, new_Nx)
        copy_Ny = min(self.Ny, new_Ny)

        new_density[:, 1:copy_Nx+1, 1:copy_Ny+1] = self.density[:, 1:copy_Nx+1, 1:copy_Ny+1]
        new_Vx[1:copy_Nx+1, 1:copy_Ny+1] = self.Vx[1:copy_Nx+1, 1:copy_Ny+1]
        new_Vy[1:copy_Nx+1, 1:copy_Ny+1] = self.Vy[1:copy_Nx+1, 1:copy_Ny+1]

        self.Nx = new_Nx
        self.Ny = new_Ny
        
        self.s = new_s
        self.density = new_density
        self.Vx = new_Vx
        self.Vy = new_Vy
        self.Vx0 = new_Vx0
        self.Vy0 = new_Vy0

    def add_source(self, grid, center_x, center_y, amount, radius=2):
        x_min = max(1, center_x - radius)
        x_max = min(self.Nx, center_x + radius)
        y_min = max(1, center_y - radius)
        y_max = min(self.Ny, center_y + radius)
        
        if x_min <= x_max and y_min <= y_max:
            x = np.arange(x_min, x_max + 1).reshape(-1, 1)
            y = np.arange(y_min, y_max + 1).reshape(1, -1)
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            grid[x_min:x_max+1, y_min:y_max+1][mask] += amount

    def step(self, dt):
        self.Vx0, self.Vx = self.Vx, self.Vx0
        self.Vy0, self.Vy = self.Vy, self.Vy0
        
        diffuse(1, self.Vx, self.Vx0, self.visc, dt, self.iter, self.Nx, self.Ny)
        diffuse(2, self.Vy, self.Vy0, self.visc, dt, self.iter, self.Nx, self.Ny)
        
        project(self.Vx, self.Vy, self.Vx0, self.Vy0, self.iter, self.Nx, self.Ny)
        
        self.Vx0, self.Vx = self.Vx, self.Vx0
        self.Vy0, self.Vy = self.Vy, self.Vy0
        
        advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, dt, self.Nx, self.Ny)
        advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, dt, self.Nx, self.Ny)
        
        project(self.Vx, self.Vy, self.Vx0, self.Vy0, self.iter, self.Nx, self.Ny)
        
        self.s, self.density = self.density, self.s
        
        for c in range(3):
            diffuse(0, self.density[c], self.s[c], self.diff, dt, self.iter, self.Nx, self.Ny)
        
        self.s, self.density = self.density, self.s
        
        for c in range(3):
            advect(0, self.density[c], self.s[c], self.Vx, self.Vy, dt, self.Nx, self.Ny)
        
        # Gradually fade out the fluid density
        decay = max(0.0, 1.0 - (0.5 * dt))
        self.density *= decay
