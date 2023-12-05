import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import seaborn as sns
import numpy

data = pd.read_csv(r'Admission.csv')
print(data)
#Extrag X si Y din csv
sns.pairplot(data, hue="Admission", diag_kind="kde")
y_real = data['Admission']
x_n = ['GRE', 'GPA']
x_real = data[x_n].values
print(data['Admission'].value_counts())

from imblearn.over_sampling import RandomOverSampler
random_over_instance = RandomOverSampler(random_state=0)
x_balanced, y_balanced = random_over_instance.fit_resample(x_real, y_real)
print(y_balanced.value_counts())

def get_model(x, y):
    with pm.Model() as model:
        alfa = pm.Normal('alfa', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))
        x_data = pm.Data('x', x)
        miu = alfa + pm.math.dot(x_data, beta)
        teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-miu)))
        bd = pm.Deterministic('bd', -alfa/beta[1] - beta[0]/beta[1] * x_data[:,0])
        yl = pm.Bernoulli('yl', p=teta, observed=y)
        idata = pm.sample(2000, target_accept=0.9, return_inferencedata=True)
    return model, idata

def show_boundary(x, y, idata, hdi_prob=None):
    idx = np.argsort(x[:,0])
    bd = idata.posterior['bd'].mean(("chain", "draw"))[idx]
    plt.scatter(x[:,0], x[:,1], c=[f'C{x}' for x in y])
    plt.plot(x[:,0][idx], bd, color='k')
    if hdi_prob is not None:
        az.plot_hdi(x[:,0], idata.posterior['bd'], color='k', hdi_prob=hdi_prob)
    else:
        az.plot_hdi(x[:,0], idata.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])
    
    return list(bd)

model_real, idata_real = get_model(x_real, y_real)

bdd = show_boundary(x_real, y_real, idata_real)

model_balanced, idata_balanced = get_model(x_balanced, y_balanced)
bd = show_boundary(x_balanced, y_balanced, idata_balanced, 0.94)


print(numpy.array(bdd).mean())

def predict_for(model, idata, x):
    with model:
        pm.set_data({'x': x})
        y_test = pm.sample_posterior_predictive(idata)
        prediction = y_test['yl'].mean()
        print(f"The rounded mean is: {round(prediction)}")
        print(f"The real mean is: {prediction}")     
        # az.plot_hdi(x, y_test['yl'])

    return prediction
def predict_for_models(x, model_balanced, idata_balanced, model_real, idata_real):
    print("Predicting with the model with balanced data ...")
    prediction_balanced = predict_for(model_balanced, idata_balanced, x)
    print()
    print("Predicting with the model with real data ...")
    prediction_real = predict_for(model_real, idata_real, x)
predict_for_models([[550, 3.5]], model_balanced, idata_balanced, model_real, idata_real)

predict_for_models([[500, 3.2]], model_balanced, idata_balanced, model_real, idata_real)    