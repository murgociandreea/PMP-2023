import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

samples = []
for _ in range(10000):
    # Alegem aleator 
    if np.random.rand() < 0.4:
        sample = stats.expon(scale=1/4).rvs()
    else:
        sample = stats.expon(scale=1/6).rvs()
    samples.append(sample)

# Media si deviatia
media_x = np.mean(samples)
deviatia_std_x = np.std(samples)

print(f'Media lui X: {media_x:.2f} ore')
print(f'Deviația standard a lui X: {deviatia_std_x:.2f} ore')
