import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

#a
if __name__ == "__main__":

    data = pd.read_csv(r'auto-mpg.csv')

    data = data.dropna(subset=['mpg', 'horsepower'])

    plt.figure(figsize=(10, 6))
    plt.scatter(data['horsepower'], data['mpg'], alpha=0.6)
    plt.title('Relația dintre Horsepower și MPG')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')

    plt.show()


#b
mpg = data['mpg'].values
horsepower = data['horsepower'].values
model =pm.Model()

with model:
        a = pm.Normal('a', mu=0, sd=10)

        b = pm.Normal('b',mu=0, sd=10)
        ε = pm.HalfCauchy('ε', 5)

        miu = pm.Deterministic('miu', a + b * horsepower)

        y_pred = pm.Normal('y_pred', mu=miu, sigma=ε, observed=mpg)

        trace = pm.sample(1000, tune=1000)

az.plot_trace(trace, var_names=['a', 'b', 'ε'])


#c

intercept_posterior = trace['intercept'].mean()
slope_posterior = trace['slope'].mean()

plt.figure(figsize=(10, 6))
plt.scatter(horsepower, mpg, alpha=0.6, label='Date observate')
plt.plot(horsepower, intercept_posterior + slope_posterior * horsepower, color='red', label='Dreapta de regresie')
plt.title('Regresie liniară între Horsepower și MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)
plt.show()


