import pandas as pd
import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

#a)Incarcam setul de date csv prin apelarea functiei pd(biblioteca pandas).read_csv
BHousing = pd.read_csv('BostonHousing.csv')
print(BHousing.head())

#b)Am extras datele din fisierul csv, am ales modelul de regresie liniara multipla

x_1 = BHousing['rm'].values
x_2 = BHousing['crim'].values
x_3 = BHousing['indus'].values
pred=BHousing['medv'].values
X = np.column_stack((x_1, x_2, x_3))  
X_mean = X.mean(axis=0, keepdims=True)
#o idee asupra mediilor si dev. standard:


with pm.Model() as model_regression:        # b
        alfa = pm.Normal('alfa', mu=0, sigma=1000)
        #am ales sigma mai mare in caz ca deviatia e mai mare
        beta = pm.Normal('beta', mu=0, sigma=1000, shape=3)
        eps = pm.HalfCauchy('eps', 5)
        X_shared = pm.MutableData('x_shared',X)
        niu = pm.Deterministic('niu', alfa+pm.math.dot(X_shared, beta))
        medv_pred = pm.Normal('medv_pred', mu=niu, sigma=eps, observed=pred)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)



#c)Estimari de 95% pentru HDI ale parametrilor; Am luat ca fiind parametri beta1, beta2, beta3 
az.plot_forest(idata,hdi_prob=0.95,var_names=['beta'])
az.summary(idata,hdi_prob=0.95,var_names=['beta'])



#d)Simularea extragerilor din distributia predictiva posterioara pt a gasi un interval de predictie de 50%
pm.set_data({"x_shared":[[random(x_1), random(x_2), random(x_3)]]}, model=model_mlr)
ppc = pm.sample_posterior_predictive(idata, model=model_mlr)
y_ppc = ppc.posterior_predictive['pred'].stack(sample=("chain", "draw")).values
az.plot_posterior(y_ppc,hdi_prob=0.5)