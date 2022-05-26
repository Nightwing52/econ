import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve
from scipy.stats import moment

#reading data
frame_sp500 = pd.read_csv("sp500_annual_returns.csv")
frame_inflation = pd.read_csv("inflation.csv")

returns = frame_sp500["value"].squeeze()
inflation = frame_inflation["value"].squeeze()

#plotting
plot_data = input("Plot historical density? (True/False) ")
print()
if plot_data == "True":
    returns.plot(kind="density")
    plt.xlabel("Annual Return")
    plt.ylabel("Density")
    plt.show()

    inflation.plot(kind="density")
    plt.xlabel("Inflation Rate")
    plt.ylabel("Density")
    plt.show()

#fitting
returns = returns.values[-1:-31:-1]
a, mean_return, std_return = skewnorm.fit(returns)
print("Return mean: "+str(mean_return)+" Return STD Dev: "+str(std_return))
print()

inflation = inflation.values[-1:-31:-1]
mean_inf, std_inf = norm.fit(inflation, loc=2.5, scale=0.5)
print("Inflation mean: "+str(mean_inf)+" Inflation STD Dev: "+str(std_inf))
print()

#plotting fit
plot_fit = input("Plot distribution fit? (True/False) ")
print()
if plot_fit == "True":
    fig, ax = plt.subplots(1,1)
    x = np.linspace(-50.0, 50.0, 100)
    returns_fit = skewnorm.pdf(x, a, mean_return, std_return)
    ax.plot(x, returns_fit)
    return_density = gaussian_kde(returns)
    ax.plot(x, return_density(x))
    plt.xlabel("Annual Return")
    plt.ylabel("Density")
    fig.show()
    plt.show()

    fig, ax = plt.subplots(1,1)
    x = np.linspace(-5.0, 15.0, 100)
    inflation_fit = norm.pdf(x, mean_inf, std_inf)
    ax.plot(x, inflation_fit)
    inf_density = gaussian_kde(inflation)
    ax.plot(x, inf_density(x))
    plt.xlabel("Inflation Rate")
    plt.ylabel("Density")
    fig.show()
    plt.show()

#simulation
n = 30 #number years
m = 10000 #number simulations
R = skewnorm.rvs(a, mean_return, std_return, size=(n, m))/100.0
I = norm.rvs(mean_inf, std_inf, size=(n, m))/100.0

results = np.prod(1+R, axis=0)*(1/np.prod(1+I, axis=0))

#plotting results
result_density = gaussian_kde(results)
mean = np.mean(results)
moments = moment(results, [1, 2, 3])
print("Mean: {:.2f}\nSTD Dev: {:.2f}\nSkew: {:.2f}\n".format(mean, moments[1], moments[2]))
cdf = lambda y : result_density.integrate_box_1d(0.0, y)-0.9
x = np.linspace(0.0, fsolve(cdf, 1.0)[0], 1000)
plt.plot(x, result_density.evaluate(x))
plt.axvline(1.0, color="r", linestyle="-")
plt.xlabel("Returns")
plt.ylabel("Density")
plt.show()

#analyzing the distribution
under_mean = result_density.integrate_box_1d(0.0, mean)
print("{:.2f}% percent of the distribution is to the left of the mean.".format(under_mean*100))