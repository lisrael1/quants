import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns
from scipy.stats import chi,gamma

r = 10000
mu = 0
sigma = 10.0

mu = 0
sigma = 1.0
r = 1000  # number of simulations, put 100000 for more accuracy
c = 10  # max number of samples per simulation

N = pd.DataFrame(np.random.normal(mu, sigma, [r, c]))
U = pd.DataFrame(np.random.uniform(-4.4, 4.4, [r, c]))
ax={}
color=['r','g','b']
print("done randoming")
for i in [10, 5, 3]:  # taking i samples out from c
    N['not_fixed_sigma_' + str(i)] = N.apply(lambda row: row[:i].std(), axis=1)
    N['fixed_sigma_' + str(i)] = N['not_fixed_sigma_' + str(i)] * i / (i - 1)
    U['not_fixed_sigma_' + str(i)] = U.apply(lambda row: row[:i].std(), axis=1)
    U['fixed_sigma_' + str(i)] = U['not_fixed_sigma_' + str(i)] * i / (i - 1)
    c=color.pop()
    ax[i]=sns.distplot(U['not_fixed_sigma_'+str(i)],kde=True, hist=False,color=c)
    sns.distplot(N['not_fixed_sigma_'+str(i)],kde=True, hist=False,color=c,ax=ax[i])
print("done calculating std")

bins = int(round(r / 50))
ax=N[[i for i in N.columns if str(i).startswith("not_fixed_sigma_")]].plot.hist(bins=bins, alpha=0.2, title="without correction",normed=True)
U[[i for i in U.columns if str(i).startswith("not_fixed_sigma_")]].plot.hist(bins=bins, alpha=0.2, title="without correction",normed=True,ax=ax)
ax=N[[i for i in N.columns if str(i).startswith("fixed_sigma_")]].plot.hist(bins=bins, alpha=0.2, title="with correction",normed=True)
U[[i for i in U.columns if str(i).startswith("fixed_sigma_")]].plot.hist(bins=bins, alpha=0.2, title="with correction",normed=True,ax=ax)
# sns.distplot(U[[i for i in U.columns if str(i).startswith("fixed_sigma_")]], fit=gamma)

# ax=sns.distplot(U['not_fixed_sigma_3'], fit=gamma,kde=False, hist=False,ax=ax)
# ax=N['not_fixed_sigma_5'].plot.hist(bins=bins, alpha=0.2, title="with correction",normed=True)
# U['not_fixed_sigma_5'].plot.hist(bins=bins, alpha=0.2, title="with correction",normed=True,ax=ax)

plt.show()

