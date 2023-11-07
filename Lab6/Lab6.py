import numpy as np
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

prior_lambda = 10

def binomial_distribution(n, theta, Y):
    return np.math.comb(n, Y) * (theta**Y) * ((1 - theta)**(n - Y))

n_values = np.arange(0, 40)

posterior_distributions = {}
for Y in Y_values:
    for theta in theta_values:
        posterior = []
        for n in n_values:
            likelihood = binomial_distribution(n, theta, Y)
            prior = np.exp(-prior_lambda) * (prior_lambda**n) / np.math.factorial(n)
            posterior.append(likelihood * prior)
        posterior /= np.sum(posterior) 
        posterior_distributions[(Y, theta)] = posterior


for Y in Y_values:
    for theta in theta_values:
        plt.plot(n_values, posterior_distributions[(Y, theta)], label=f"Y={Y}, θ={theta}")

plt.xlabel('n')
plt.ylabel('Probabilitatea a posteriori')
plt.legend()
plt.show()