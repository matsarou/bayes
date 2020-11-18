import math
import time

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO, NUTS, MCMC, HMC
from torch.distributions import constraints
import seaborn as sns
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

def main():
    start = time.time()
    pyro.clear_param_store()

    # the kernel we will use
    hmc_kernel = HMC(model, step_size=0.1)

    # the sampler which will run the kernel
    mcmc = MCMC(hmc_kernel, num_samples=14000, warmup_steps=100)

    # the .run method accepts as parameter the same parameters our model function uses
    mcmc.run(data)
    end = time.time()
    print('Time taken ', end - start, ' seconds')

    sample_dict = mcmc.get_samples(num_samples=5000)

    plt.figure(figsize=(10, 7))
    sns.distplot(sample_dict['latent_fairness'].numpy(), color="orange");
    plt.xlabel("Observed probability value")
    plt.ylabel("Observed frequency")
    plt.show()

    mcmc.summary(prob=0.95)



if __name__== "__main__":
    main()