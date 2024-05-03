import random
import math
import matplotlib.pyplot as plt

binomial = lambda n, k: math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

prior = 0.5
distribution = [prior for _ in range(10)]

n = 10
ones = sum([1 if random.random() > 0.5 else 0 for _ in range(n)])

for i in range(10):
    distribution = [prior * (
                    (binomial(n, ones) * (p**ones) * ((1-p)**(n-ones)))
                    /
                    (binomial(n, ones) * (0.5**ones) * (0.5**(n-ones)))
                    ) for p in distribution]

print(distribution)
plt.bar(range(len(distribution)), distribution)
plt.show()

# now lets update the distribution
# we update our prior
# prior(θ) * (f(1|θ) / f(1))
# f(1|θ) = binomial(n, ones) * θ^ones * (1-θ)^(n-ones)
# f(1) = binomial(n, ones) * (0.5**ones) * (0.5**(n-ones))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Initial parameters for the Beta distribution (prior)
alpha_prior = 1
beta_prior = 1

# Number of random experiments
num_experiments = 10

# Simulate 10 random experiments with outcomes 0 or 1
for i in range(num_experiments):
    data = np.random.choice([0, 1], size=num_experiments)
    data = [1 for _ in range(10)]

    # Update the distribution based on the observed data
    alpha_posterior = alpha_prior + np.sum(data)
    beta_posterior = beta_prior + num_experiments - np.sum(data)

print(f"Prior: alpha={alpha_posterior}, beta={beta_posterior}")

# Generate a range of x values for plotting the prior and posterior distributions
x = np.linspace(0, 1, 1000)

# Calculate the prior and posterior distributions
prior_distribution = beta.pdf(x, alpha_prior, beta_prior)
posterior_distribution = beta.pdf(x, alpha_posterior, beta_posterior)

# Plot the prior and posterior distributions
plt.plot(x, prior_distribution, label='Prior Distribution', linestyle='--')
plt.plot(x, posterior_distribution, label='Posterior Distribution', linestyle='-')

plt.title('Bayesian Inference with Bayes Factor')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend()
plt.show()

