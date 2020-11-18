import pandas as pd
import pyro
import pyro.distributions as dist
import seaborn as sns


def find_weather():
    # The variable weather is associated to a primitive stochastic function
    weather = pyro.sample('weather', pyro.distributions.Bernoulli(0.3))
    # convert 0 or 1 to cloudy or sunny
    weather = 'cloudy' if weather.item() == 1.0 else 'sunny'

    # subset corresponding mean temp out of a dictionary
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[weather]
    var_temp = {'cloudy': 10.0, 'sunny': 15.0}[weather]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, var_temp))

    return weather, temp.item() # return the weather and temp

# Condition the model weather on observed data and infer the latent factors that might have produced that data
def find_weather(obs = 40):
    # The variable weather is associated to a primitive stochastic function
    weather = pyro.sample('weather', pyro.distributions.Bernoulli(0.5))
    # convert 0 or 1 to cloudy or sunny
    weather = 'cloudy' if weather.item() == 1.0 else 'sunny'

    # subset corresponding mean temp out of a dictionary
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[weather]
    var_temp = {'cloudy': 10.0, 'sunny': 15.0}[weather]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, var_temp), obs=obs)

    return weather, temp # return the weather and temp

if __name__== "__main__":
    data = [find_weather()[1] for i in range(1000)]
    sns.distplot(data)
    # plt.show()

    #write the weather model once and condition it on one observation.
    data = [find_weather(40) for i in range(1000)]
    df = pd.DataFrame(data, columns=('weather', 'temp'))
    print(df.head())