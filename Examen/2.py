import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#Am modificat probabilitatea(likelihood) cu stats.geom.pmf
def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) 
    likelihood = stats.geom.pmf(np.arange(1, len(data)+1), grid)  
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

aruncari = np.random.choice([0, 1], size=100)  
prima_stema_index = np.where(aruncari == 1)[0][0]
grid, posterior = posterior_grid(aruncari)