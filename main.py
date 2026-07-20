import sys
import pygame
import colorsys
import numpy as np

from fluid_engine import Fluid

# --- Main Simulation Loop ---
# Handles window setup, user input, and rendering the fluid grid.
def main():
    SCALE = 4
    N_INITIAL = 200
    
    current_w = N_INITIAL * SCALE
    current_h = N_INITIAL * SCALE
    
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 20)
    
    screen = pygame.display.set_mode((current_w, current_h), pygame.RESIZABLE)
    pygame.display.set_caption("Optimized Numba Eulerian Fluid Dynamics")
    clock = pygame.time.Clock()
    
    fluid = Fluid(current_w // SCALE, current_h // SCALE, iter_count=16, diff=0.0, visc=0.0001)
    
    running = True
    prev_mouse = None
    
    frames = 0
    hue = 0.0
    
    print("Starting fluid simulation...")
    
    while running:
        # Calculate time passed since last frame (capped to avoid instability)
        raw_dt = clock.tick(120) / 1000.0
        dt = max(0.016, min(raw_dt, 0.05))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                current_w, current_h = event.w, event.h
                screen = pygame.display.set_mode((current_w, current_h), pygame.RESIZABLE)
                
                SCALE = 4 
                fluid.resize(current_w // SCALE, current_h // SCALE)
            elif event.type == pygame.KEYDOWN:
                # Dynamic parameter controls
                if event.key == pygame.K_UP:
                    fluid.visc = min(fluid.visc + 0.0001, 0.01)
                elif event.key == pygame.K_DOWN:
                    fluid.visc = max(fluid.visc - 0.0001, 0.0)
                elif event.key == pygame.K_RIGHT:
                    fluid.diff = min(fluid.diff + 0.0001, 0.01)
                elif event.key == pygame.K_LEFT:
                    fluid.diff = max(fluid.diff - 0.0001, 0.0)
                
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        
        hue = (hue + 0.1 * dt) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        
        if mouse_pressed[0]: 
            grid_x = int(mouse_pos[0] / SCALE)
            grid_y = int(mouse_pos[1] / SCALE)
            
            if prev_mouse is not None:
                dx = mouse_pos[0] - prev_mouse[0]
                dy = mouse_pos[1] - prev_mouse[1]
                
                # Inject velocity into the fluid
                speed_scale = 10.0
                fluid.add_source(fluid.Vx, grid_x, grid_y, dx * dt * speed_scale, radius=4)
                fluid.add_source(fluid.Vy, grid_x, grid_y, dy * dt * speed_scale, radius=4)
                
            fluid.add_source(fluid.density[0], grid_x, grid_y, r * 15000.0 * dt, radius=4)
            fluid.add_source(fluid.density[1], grid_x, grid_y, g * 15000.0 * dt, radius=4)
            fluid.add_source(fluid.density[2], grid_x, grid_y, b * 15000.0 * dt, radius=4)
            prev_mouse = mouse_pos
        else:
            prev_mouse = None
            
        fluid.step(dt)
        
        d_cpu = np.clip(fluid.density[:, 1:-1, 1:-1], 0, 255).astype(np.uint8)
        
        Nx, Ny = fluid.Nx, fluid.Ny
        rgb = np.zeros((Nx, Ny, 3), dtype=np.uint8)
        rgb[:, :, 0] = d_cpu[0]
        rgb[:, :, 1] = d_cpu[1]
        rgb[:, :, 2] = d_cpu[2]
        
        surf = pygame.surfarray.make_surface(rgb)
        pygame.transform.scale(surf, (current_w, current_h), screen)
        
        frames += 1
        
        # Draw statistics
        stats = [
            f"FPS: {int(clock.get_fps())}",
            f"Grid: {Nx}x{Ny}",
            f"Scale: {SCALE}x",
            f"Viscosity (UP/DOWN): {fluid.visc:.4f}",
            f"Diffusion (LEFT/RIGHT): {fluid.diff:.4f}"
        ]
        for i, stat in enumerate(stats):
            text_surf = font.render(stat, True, (255, 255, 255))
            screen.blit(text_surf, (10, 10 + i * 20))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
