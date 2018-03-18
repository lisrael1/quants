import pandas as pd
import numpy as np
import pylab as plt


mu = 100
sigma = 10
r = 10000 # number of simulations
c = 50 # number of samples per simulation
# df = pd.DataFrame(np.random.normal(mu, sigma, [r,c]))
df = pd.DataFrame(np.random.uniform(-10, 10, [r,c]))
print("done generating")
for i in [5,10,30,50]:
    df['sigma_'+str(i)] = df.apply(lambda row : np.std(row[:i]),axis=1)

print("done calc std")
df=df[[i for i in df.columns if "sigma" in str(i) ]]
df.plot.hist(bins=1000,alpha=0.2)
print("done doing histogram")
plt.show()
exit()


def mod(numbers, modulo_size_edge_to_edge):
    '''
        modulo_size_edge_to_edge should be array of size 1 or at numbers column size, for example:
            mod(numbers,[6.0]
            mod(numbers,[6.0,100])#for mod 6 for the first column and 100 for the second column
        modulo_size_edge_to_edge should be numpy matrix
        TODO add modulo size 0 for disabling modulo - just remove column and return it when finishing
    '''
    num_resize = (numbers + modulo_size_edge_to_edge / 2.0) % modulo_size_edge_to_edge
    num_resize -= modulo_size_edge_to_edge / 2.0
    return num_resize


mu = 0
# sigma = 1
s = 500

for i in range(10):
    # for sigma in [0.1,0.5,1,2]:
    sigma = 0.2
    n1 = np.random.normal(mu, sigma, s).round(3)
    n2 = mod(n1, 2)
    pd.DataFrame(n2).hist()
    plt.show()
    print(np.std(n2))


