from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class BetaDistribution:

    def __init__(self):
        self.a = 1
        self.b = 1
        self.n = 0
        self.data = []
        # self.span = np.linspace(0, 1, 200)
        self.span = np.linspace(self.ppf(0.01), self.ppf(0.99), 200)

    def update_params(self, data = []):
        self.data = self.data +data
        self.n = len(self.data)
        x = sum(data)
        # self.a = self.parameter_a()
        # self.b = self.parameter_b()
        self.a += x
        self.b += self.n - x

    def mean(self):
        tr = sum(self.data)
        fl = self.n -tr
        return tr * self.n / (tr + fl)
        # return stats.moment(self.data, 1) * self.n

    def stdev(self):
        return stats.moment(self.data, 2) * self.n

    def pdf(self):
        return stats.beta.pdf(self.span, self.a, self.b)

    def ppf(self, x):
        return stats.beta.ppf(x, self.a, self.b)

    #  computes a binomial cumulative distribution function at each of the values in x using the corresponding number
    #  of trials in n and the probability of success for each trial in p.
    def cdf(self, loc=0, scale=1):
        return stats.beta.cdf(self.span, self.a, self.b, loc, scale)

    def parameter_a(self):
        return self.mean()
        # b = self.parameter_b()
        # mean = self.mean()
        # return b * mean / (1 - mean)

    def parameter_b(self):
        return self.stdev()
        # mean = self.mean()
        # stdev = self.stdev()
        # return mean - 1 + mean * pow(1 - mean, 2) / pow(stdev, 2)

    # The z-score is 1.96 for a 95% confidence interval.
    def conf_interval(self, z_score = 1.96):
        # The size of the successes
        mean = self.mean()
        # standard error
        se = np.sqrt(mean * (1 - mean) / self.n)
        # the confidence interval
        lcb = mean - z_score * se  # lower limit of the CI
        ucb = mean + z_score * se  # upper limit of the CI
        return lcb, ucb

    def mean_confidence_interval(self, confidence=0.95):
        n = len(self.data)
        m = np.mean(self.data)
        std_err = stats.sem(self.data)
        h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return m - h, m + h

    def show_plot(self, params):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.span, self.pdf(), alpha = 1.0, color=params['color'], label=params['label'])
        if self.mean() > 0.0 and self.n > 0:
            lcb, ucb = self.mean_confidence_interval()
            ax.fill_between(self.data, lcb, ucb, color='b', alpha=.1)
        plt.ylabel('density')
        plt.xlabel('conversion rate')
        plt.legend(numpoints=1, loc='upper right')
        plt.show()
