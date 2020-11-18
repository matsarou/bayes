import numpy as np
from scipy.stats import norm, beta, binom
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns

data=[]
# Collect some data.
# - 1 for success
# - 0 for failure
for i in range(0, 6):
    data.append(1.0)
for i in range(0, 4):
    data.append(0.0)

def beta_pdf(p, a, b):
    return beta(a, b).pdf(p)

def logBinomial_pmf(k, n, p):
    # The statistical distribution that represents the data generative process is a binomial distribution. pmf
    # parameters:
    # n trials
    # θ∈[0,1]
    return binom.logpmf(k, n, p)

def binomial_pmf(k, n, p):
    # The statistical distribution that represents the data generative process is a binomial distribution. pmf
    # parameters:
    # n trials
    # θ∈[0,1]
    return binom.pmf(k, n, p)

def likelihood_product(data, p, n, pmf):
    if (p < 0 or p > 1):
        return 0
    else:
        return pmf(data, p, n)

def normal(loc, scale):
    return norm(loc, scale)

# Integrate input 'y' along given axis 'x'
def normalize(y, x):
	return y/np.trapz(y,x)

# k: number of successes
# n: numbers of trials
# alpha, beta: parameters of beta distribution
def binomial_model(k, n, alpha, beta):
    theta = np.linspace(0, 1, 201)
    prior = beta_pdf(theta, alpha, beta)
    posterior = beta_pdf(theta, alpha+k, beta+n-k)
    likelihood = [binomial_pmf(k, n, p) for p in theta]
    likelihood = normalize(likelihood, theta) # integrate likelihood along theta, for display reasons

    plt.figure(figsize=(16, 4))
    plt.plot(theta, prior, label='Prior', linestyle='--')
    plt.plot(theta, posterior, label='Posterior')
    plt.plot(theta, likelihood, label='Likelihood')
    plt.xlim((0, 1))
    plt.legend()
    plt.show()

    return (theta, posterior)

# binomial_model(7,10,1,1)
# binomial_model(7,10,2,2)
binomial_model(7,10,20, 20)

# n=201
# a, b = 1, 1 #same thetas(p) fpr all values.
# p = np.linspace(0,1,n)
# posteriors = []
# n_samples = [20, 50, 100, 500, 5000]
# for i, N in enumerate(n_samples):
#     bion_trials = binom(N, p).rvs(size=N)
#     prior_model = beta_pdf(p, a, b)  # Uninformative prior
#     likelihood_current = likelihood_product(bion_trials, prior_model, N, logBinomial_pmf)
#
#     posterior = beta_pdf(prior_model, a, b)
#     posteriors.append(posterior)

