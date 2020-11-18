import math
import time

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO, NUTS, MCMC, HMC
from torch.distributions import constraints
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pyro.optim import Adam #the problem is on Pycharm

# Question: “What is the probability
# that any famous person (like Shaq) can drop by the White
# House without an appointment?”

pyro.set_rng_seed(101)

data=[]
# Collect some data.
# - 1 for success
# - 0 for failure
for i in range(0, 6):
    data.append(torch.tensor(1.0))
for i in range(0, 4):
    data.append(torch.tensor(0.0))

pyro.clear_param_store()

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    f = pyro.sample("latent_fairness", dist.Beta(alpha0,beta0))

    # Determine the binomial likelihood of the observed data, assuming each hypothesis is true
    for i in range(len(data)):
        pyro.sample(f'obs_{i}',
                             dist.Binomial(probs=f),
                             obs=data[i])

def get_proposal(mean_current, std_current, proposal_width = 0.5):
    pyro.sample("proposedDist", dist.Normal(std_current, proposal_width))




def model_proposed(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    f = pyro.sample("prior_proposed", dist.Beta(alpha0,beta0))

    # Determine the binomial likelihood of the observed data, assuming each hypothesis is true
    for i in range(len(data)):
        pyro.sample(f'obs_prop_{i}',
                             dist.Binomial(probs=f),
                             obs=data[i])

def model_current(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    f = pyro.sample("prior_current", dist.Beta(alpha0,beta0))

    # Determine the binomial likelihood of the observed data, assuming each hypothesis is true
    for i in range(len(data)):
        pyro.sample(f'obs_prop_{i}',
                             dist.Binomial(probs=f),
                             obs=data[i])



def main():
    start = time.time()
    pyro.clear_param_store()

    end = time.time()
    print('Time taken ', end - start, ' seconds')

    observation = torch.stack(data).numpy()
    # observation = observation[np.random.randint(0, 30000, 1000)]


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(observation, bins=35, )
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.show()



if __name__== "__main__":
    main()