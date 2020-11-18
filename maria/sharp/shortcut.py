import random
import numpy as np
import matplotlib as ml
from scipy.stats import poisson
from scipy.stats import gamma
import matplotlib.pyplot as plot

# https://github.com/jvans1/bayesian_machine_learning/blob/master/Conjugate%20Priors.ipynb

def model():
    µ = 4
    x = np.arange(poisson.ppf(0.01, µ), poisson.ppf(0.99, µ))
    y = poisson.pmf(x, µ)
    prior_hist = plot.hist(x, np.arange(0, 10), weights=y, align="left", rwidth=0.8)

def confident_prior():
    prior_x, prior_y = gamma(scale= 1, prior_shape = 35)
    plot(prior_x, prior_y)
    return prior_x, prior_y

def uncertain_prior():
    prior_x, prior_y = gamma(scale= 1, prior_shape = 4)
    plot(prior_x, prior_y)
    return prior_x, prior_y

def posteriorr():
    posterior_x, posterior_y = gamma(scale=0.5, prior_shape=10)
    return posterior_x, posterior_y


def conjugacy_solution():
    prior_x, prior_y = confident_prior()
    posterior_x, posterior_y = posteriorr()
    # plot
    fig, ax = plot.subplots(1, 1)
    posterior, = ax.plot(posterior_x, posterior_y, 'b-', label="posterior")
    prior, = ax.plot(prior_x, prior_y, 'r-', label="prior")
    ax.legend(handles=[posterior, prior])


def gamma(scale, prior_shape):
    x = np.linspace(gamma.ppf(0.001, prior_shape, loc=0, scale=scale),
                          gamma.ppf(0.999, prior_shape, loc=0, scale=scale))
    y = gamma.pdf(x, prior_shape, loc=0, scale=scale)
    return x, y

def plot(prior_x, prior_y):
    fig, ax = plot.subplots(1, 1)
    confident_prior, = ax.plot(prior_x, prior_y, 'r-', label="prior")

    ax.plot(prior_x, prior_y, 'r-', label='gamma pdf')
    ax.legend(handles=[confident_prior])


mu = 0.6
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
print("mean={0}, var={1}, skew={2}, kurt={3}".format(mean, var, skew, kurt))

x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
print(x)