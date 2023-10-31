import pandas as pd
import pymc3 as pm
import numpy as np

data = pd.read_csv('trafic.csv')
minute = data['minut'].values
nr_masini = data['nr_masini'].values
# Ora la care au loc modificările mediei traficului
ore_modificari = [7, 16, 8, 19]

with pm.Model() as model:
    
    lambda_prior = pm.Exponential('lambda', lam=1.0)
    
    lambda_ = lambda_prior * 1.5  # Creștere la ora 7
    lambda_ = pm.math.switch(ore_modificari[2] <= pm.math.arange(len(nr_masini)), lambda_prior /1.5, lambda_) 
    lambda_ = pm.math.switch(ore_modificari[1] <= pm.math.arange(len(nr_masini)), lambda_prior *1.5, lambda_)  
    lambda_ = pm.math.switch(ore_modificari[3] <= pm.math.arange(len(nr_masini)), lambda_prior /1.5, lambda_)  
    observatii = pm.Poisson('observatii', mu=lambda_, observed=nr_masini)
with model:
    trace = pm.sample(2000, tune=1000, target_accept=0.9)


#2
intervale = [(0, 7), (7, 16), (16, 19), (19, 24)]
intervale_probabile = {}

for interval in intervale:
    ora_inceput, ora_sfarsit = interval
    minute_inceput = (ora_inceput - 4) * 60  
    minute_sfarsit = (ora_sfarsit - 4) * 60  
