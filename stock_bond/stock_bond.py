import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import moment

mu = np.array([9.87, 2.31, 2.0])
C = np.array([[284.80, -4.34, 0.0], [-4.34, 1.02, 1.10], [0.0, 1.10, 3.51]])

#runs a simulation; takes stock allocation and parameters; returns median, STD deviation, and skew
def sim(stock_alloc, plot = False):
    n = 30 #number years
    m = 10000 #number simulations
    RV = np.random.multivariate_normal(mu, C, n*m) #drawing samples
    RV = 1.0+RV/100.0 #scaling

    SR, BR, I = np.prod(np.reshape(RV[:, 0], (n, m)), axis=0), np.prod(np.reshape(RV[:, 1], (n, m)), axis=0), np.prod(np.reshape(RV[:, 2], (n, m)), axis=0)

    results = (stock_alloc*SR+(1.0-stock_alloc)*BR)/I 
    
    #plot if needed
    if plot:
        plt.hist(results, bins=300, density=True)
        plt.xlabel("Returns")
        plt.ylabel("Density")
        plt.xlim(0.0, 20.0)
        plt.show()

    std = np.std(results)
    skew = moment(results, moment=3)/std**3
    return [np.median(results), std, skew]

#loading data
df = pd.read_csv("data.csv")
df = df.loc[28:57]

stock = df["market"].squeeze()
bond = df["bond"].squeeze()
inflation = df["inflation"].squeeze()

#coorelation matrix
corr_matrix = df.corr()
print(corr_matrix)
print(df.cov())
print(df.mean())
if input("Plot scatter matrix? (True/False) ") == "True":
    pd.plotting.scatter_matrix(df, diagonal="kde")
    plt.show()

#plotting data
if input("Plot historical density? (True/False) ") == "True":
    bond.plot(kind="hist", bins=10)
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.show()

#fitting
bond_par = norm.fit(bond)

bond_fit = norm.rvs(bond_par[0], bond_par[1], size=100)

plot_data = pd.DataFrame([bond, bond_fit])
plot_data = plot_data.transpose()
plot_data.columns = ["Bond", "Bond Fit"]

#plotting fit
if input("Plot bond fit? (True/False) ") == "True":
    plot_data[["Bond", "Bond Fit"]].plot.kde(bw_method=0.9)
    plt.show()

#simulation
print(sim(0.6))

#plotting stock allocation vs. median returns 
x = np.linspace(0.0, 1.0, 100)
y = np.array([sim(x_i) for x_i in x])
plt.plot(x, y[:, 0])
plt.xlabel("Stock Allocation")
plt.ylabel("Returns")
plt.show()
plt.plot(x, y[:, 1])
plt.xlabel("Stock Allocation")
plt.ylabel("Volatility")
plt.show()

#Sharpe ratio
plt.plot(x, (y[:, 0]-y[0, 0])/y[:, 1])
plt.xlabel("Stock Allocation")
plt.ylabel("Sharpe Ratio")
plt.show()