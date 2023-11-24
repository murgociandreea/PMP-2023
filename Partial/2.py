import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

# Generăm date de antrenare folosind o distribuție normală cu parametrii a priori
np.random.seed(50)  # Pentru reproducibilitate
alfa_prior = 1.0    # Parametrul alfa pentru distribuția a priori
miu_prior = 10.0    # Parametrul miu pentru distribuția a priori
timpi_medii_asteptare = np.random.normal(loc=miu_prior, scale=alfa_prior, size=200)

