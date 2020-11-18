import numpy as np
from scipy.stats import norm, beta, binom

data=[]
# Collect some data.
# - 1 for success
# - 0 for failure
for i in range(0, 6):
    data.append(1.0)
for i in range(0, 4):
    data.append(0.0)

def beta_pdf(p, a, b):
    # pmf
    if (p < 0 or p > 1):
        return 0
    return beta(a, b).pdf(p)

def logBinomial_pmf(data, p, n):
    # The statistical distribution that represents the data generative process is a binomial distribution. pmf
    # parameters:
    # n trials
    # θ∈[0,1]
    return binom.logpmf(data, n, p)

def likelihood_product(data, p, n, pmf):
    if (p < 0 or p > 1):
        return 0
    else:
        return pmf(data, p, n).prod()

def normal(loc, scale):
    return norm(loc, scale)

def sampler(data, mu_init=.5, proposal_width=.5, plot=False):
    # indicators for the weights of the values of p
    a, b = 1, 1
    n = 20

    # Find starting parameter position (can be randomly chosen), we fix it arbitrarily to mu_init
    mu_current = mu_init

    n_iter = 100
    # Propose to move (jump) from that position somewhere else
    # needs to be symmetric(Normal)
    posterior = [mu_current]
    # Repeat the resampling procedure many times (here, 10000)
    for i in range(n_iter):
        # Propose to move from the current position to a new position(that's the Markov part).
        # The Metropolis sampler is very dumb and just takes a sample from a normal distribution, centered around your current mu value (i.e.mu_current)
        # with a certain standard deviation (proposal_width) that will determine how far you propose jumps
        mu_proposal = normal(mu_current, proposal_width).rvs()

        #  Determine the likelihood of the observed data, assuming each hypothesis for λ is true.
        # The likelihood helps to evaluate how good our model explains the data.
        #  Compute likelihood by multiplying probabilities of each data point
        likelihood_current = likelihood_product(data, mu_current, n, logBinomial_pmf)
        likelihood_proposal = likelihood_product(data, mu_proposal, n, logBinomial_pmf)

        # Compute prior probability of current and proposed mu.
        # Prior is what our belief about μ is before seeing the data.
        # Use a non-informative prior distribution. Set equal weight on all values of mu, where α = 1 and b = 1
        prior_current = beta_pdf(mu_current,a, b)
        prior_proposal = beta_pdf(mu_proposal, a, b)

        # Nominator of Bayes formula
        p_current = likelihood_current + prior_current
        p_proposal = likelihood_proposal + prior_proposal

        # Accept proposal?
        # Evaluate whether the new position is a good place to jump to or not. If the resulting posterior distribution with that proposed mu explains
        # the data better than your old mu, you'll definitely want to go there.
        p_accept = p_proposal / p_current
        accept = np.random.rand() < p_accept

        # if plot:
            # plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i)

        if accept:
            # Update position
            mu_current = mu_proposal

        posterior.append(mu_current)

    return np.array(posterior)

sampler(data, mu_init=0.2, plot=True);