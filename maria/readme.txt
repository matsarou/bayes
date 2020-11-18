Question we try to answer
"What is the probability p, that any famous person (like Shaq) can drop by the White House without an invitation?”

Step 1
Since p is a continuous variable, we consider the full range of hypotheses between 0 and 1, which are infinite.
(p = 0.01, p = 0.011, p = 0.0111, etc.).

Step 2
The number of p values is infinite in [0,1], so we use a pdf to represent the prior.
We set up a prior distribution (a beta distribution) defining the parameters - (α0,β0)

Step 3
We collect some data (in the form of successful or failed attempts to visit the White House without an invitation). Binomial distribution

Step 4
Determine the likelihood of the observed data

Step 5
We update to the posterior beta distribution with an analytical (conjugate) shortcut that allows us to avoid the integration in the denominator of Bayes’ Theorem.
The integration of the denominator is sometimes impossible.
Something else concerning is that the posterior calculation gets expensive.
First, we have to calculate the posterior for thousands of thetas in order to normalize it.
Second, if there is no closed-form formula of the posterior distribution, we have to find the maximum posterior by numerical optimization, such as gradient descent or newtons method.

Conjugate priors
For problems where the prior distribution is a beta distribution(α0,β0), and the data collected come in the form of binomial data (number of successes out of a given number of trials),
an analytical approach can be used to generate the posterior distribution.
In the "White house" example, the beta distribution is a conjugate prior to the binomial likelihood.
What does this mean? It means during the modeling phase, we already know the posterior will also be a beta distribution. With a conjugate prior, we can skip the
posterior = likelihood * prior computation and still find the maximum without normalizing.
However, if we want to compare posteriors from different models, or calculate the point estimates, we need to normalize.
By using a conjugate prior, we generate our posterior by updating the parameters of our prior — reflecting a new mean and confidence level, depending on the amount of observed data.
As we observe more datapoints, 𝑘 and 𝜃 are updated in such a way as to shrink the width of our posterior, indicating an increased level of confidence in our distribution.
The analytical shortcut makes updating to the posterior a snap. Here it is:
• posterior α = α0 + y
• posterior β = β0 + n − y

However, not all problems can be solved this way . . . Bayesian conjugates are special cases that can be solved analytically.
But, there is another approach, one that can be used to solve almost any kind of parameter estimation problem.
It involves building the posterior distribution from scratch using a process called a Markov Chain Monte Carlo simulation, or MCMC for short.
Another problem with the MCMC is that if we increase the dimensions, or define a more complex model, then the calculation of P(D) becomes intractable.

Markov Chain Monte Carlo (MCMC) approach
MCMC is a stochastic process
The normalization does not change the relative probabilities of θ. That is because the denominator is fixed across all hypotheses.
This insight means that the posterior density of a given hypothesis is "proportional to" the likelihood of observing the data under the
hypothesis(joint probability of P(θ,D)) times the prior density of the given hypothesis.
P(θ|D)=∝P(D|θ)P(θ)

So we don’t have to compute the evidence P(D) to infer which parameters are more likely!
However, making a grid over all of θ with a reasonable interval (which is what we did in our example), is still very expensive.
It turns out that we don’t need to compute P(D|θ)P(θ) for every possible θi , but that we can sample θi proportional to the probability mass.
This is done by exploring θ space by taking a random walk and computing the joint probability P(θ,D) and saving the parameter sample of θi
according to the following probability:
P(acceptance) = min(1, P(D|θ*)P(θ*) / P(D|θ)P(θ)
Where θ= current state, θ∗= proposal state.


Inference