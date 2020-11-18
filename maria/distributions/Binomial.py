from scipy.stats import binom

# x: number of successes
# n: number of trials
# p: probability of success
#  computes a binomial cumulative distribution function at each of the values in x using the corresponding number
#  of trials in n and the probability of success for each trial in p.
def cdf(x, n, p):
    return binom.cdf(x, n, p)