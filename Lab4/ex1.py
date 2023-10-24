import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import pymc3 as pm
import arviz as az


# problem 1

model = pm.Model()
with model:
    alpha = 4 
    clienti_ora = pm.Poisson('clienti_ora' ,alpha)
    timp_asteptare = pm.Normal('timp_asteptare', 1, sigma=1/2)
    medie_comanda = pm.Exponential('medie_comanda', 1/alpha)
    trace = pm.sample(20)

dictionary = {
              'timp_preparare': trace['medie_comanda'].tolist(),
              'timp_asteptare': trace['timp_asteptare'].tolist(),
              }
df = pd.DataFrame(dictionary)

# problem 2
timp_sub15 = df[(df['timp_asteptare'] + df['timp_preparare'] <= 15)]

sample_size =  df.shape[0]

timp_sub15procente = timp_sub15.shape[0] / sample_size
print("Waiting time under 15 minutes percentage", timp_sub15procente)

# problem 3
asteptare_client = list(sum(x) for x in zip(df['timp_asteptare'], df['timp_preparare']))
medie_asteptare_client = sum(asteptare_client) / sample_size

print("Average customer waiting time:", medie_asteptare_client)