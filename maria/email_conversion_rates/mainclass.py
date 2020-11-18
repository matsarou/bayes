from numpy import random

from maria.distributions.BetaDistribution import BetaDistribution
import numpy as np

def data_prob(prob = 0.20, size = 100):
    return [1 if i <= prob else 0 for i in np.linspace(0,1,size)]

beta = BetaDistribution()

# plot the prior with an initial likelihood
data = data_prob(0.4, 5)
beta.update_params(data)
beta.show_plot({
    'color': 'b',
    'alpha': 0.4,
    'label': 'a=' + str(beta.a) + ', b=' + str(beta.b)
})

# Collect data
data = data_prob(0.2, 500) + data_prob(0.4, 500) + data_prob(0.6, 500) + data_prob(0.8, 500)

# Inference
size = len(data)
window = int(size / 4)
batches = [batch for batch in range(0, size + window, window)]
colors = ['g', 'y', 'r', 'o']
for i in range(1, len(batches)):
    curr = batches[i]
    prev = batches[i - 1]
    batch = data[prev:curr]
    # a = sum(batch)
    # b = len(batch) - a
    beta.update_params(batch)
    dict = {
        'color': colors[i-1],
        'alpha': 0.4,
        'label': 'a=' + str(beta.a) + ', b=' + str(beta.b)
    }
    beta.show_plot(dict)
