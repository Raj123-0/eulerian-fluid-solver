import cProfile
import pstats
import sys
import os

sys.path.append(r'c:\Users\davea\OneDrive\Desktop\Eulerian Fluid Dynamics Solver')
import importlib.util

spec = importlib.util.spec_from_file_location("eulerian", r"c:\Users\davea\OneDrive\Desktop\Eulerian Fluid Dynamics Solver\Eulerian Fluid Dynamics Solver.py")
eulerian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eulerian)

fluid = eulerian.Fluid(100)

profiler = cProfile.Profile()
profiler.enable()
for _ in range(50):
    fluid.step()
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)
