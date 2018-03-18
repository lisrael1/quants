import pandas as pd
import numpy as np
import pylab as plt
import cufflinks
import plotly as py

import seaborn as sns

r = 1000  # number of simulations, put 100000 for more accuracy
c = 30  # max number of samples per simulation

U = pd.DataFrame(index=range(r))
U['del'] = 0
color = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
print("done randoming")
for i in [200, 30, 10, 4, 3, 2]:
    U['std from %02d samples' % i] = U.apply(lambda dummy: np.random.uniform(0, 1, i).std(), axis=1)
    c = color.pop()
    sns.distplot(U['std from %02d samples' % i], kde=True, hist=False, color=c, label='%02d samples' % i, axlabel='std value')
print("done calculating std")
U.drop('del', inplace=True, axis=1)
plt.suptitle('std from sampled U')

if 1:
    plt.show()
else:
    fig = U.iplot(kind='hist', histnorm='probability density', asFigure=True)
    py.offline.plot(fig, filename='a.html')
