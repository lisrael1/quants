import pandas as pd
import numpy as np
import pylab as plt
# import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})



A=np.mat([[1,2],
          [-3,-4]])
# cov = [[10,9.5],[9.5,10]]
cov=A.T.I*(A.I)
modulo_size_edge_to_edge=8
samples=1000





def random_data(cov,samples):
    xy=pd.DataFrame(np.random.multivariate_normal([0,0], cov, samples),columns=['X','Y'])
    return xy
def sign_mod(xy,modulo_size_edge_to_edge):
    xy=xy.copy()
    xy+=modulo_size_edge_to_edge/2.0
    xy=xy.mod(modulo_size_edge_to_edge)-modulo_size_edge_to_edge/2.0
    xy.columns=['X','Y']
    return xy
def quantize(xy,modulo_size_edge_to_edge,number_of_bins):
    hlf=modulo_size_edge_to_edge/2.0
    bins = np.linspace(-hlf, hlf, number_of_bins+1)
    center = (bins[:-1] + bins[1:]) / 2  # which is also (q[:-1]+(q[1]-q[0])/2)
    bins[0] = -float("inf")  # otherwise the values outside the bins will get NaN
    bins[-1] = float("inf")
    df=pd.DataFrame()
    df['X'] = pd.cut(xy.X, bins, labels=center).astype(float)
    df['Y'] = pd.cut(xy.Y, bins, labels=center).astype(float)
    return df
def plot(data,title):
    pd.DataFrame(data,columns=['X','Y']).plot.scatter(x='X', y='Y', title=title, ax=axes.pop(),alpha=0.2,grid=True)
    # pd.DataFrame(data,columns=['X','Y']).plot.hexbin(x='X', y='Y', title=title,gridsize=10, ax=axes.pop())
    # sns.jointplot(x="X", y="Y", data=df_mod, kind="kde")
def plot_bars(df):
    # pd.plotting.table(axes.pop(),mse, loc='center')
    # sns.heatmap(xy_mse, annot=True,ax=axes.pop())
    df=xy_mse.copy().round(2)
    ax=df.plot.bar(ax=axes.pop(), title="6.MSE",alpha=0.6)
    ax.annotate(float(df.X), xy=(0, df.X),horizontalalignment='right')
    ax.annotate(float(df.Y), xy=(0, df.Y),horizontalalignment='left')





mse_all_bins=pd.DataFrame()
i=0
for bins in range(3,30):
    df_original=random_data(cov,samples)
    df_mod1=sign_mod(df_original,modulo_size_edge_to_edge)
    df_quant=quantize(df_mod1,modulo_size_edge_to_edge,bins)
    df_A=df_quant.dot(A)
    df_A.columns=['X','Y']
    df_mod2=sign_mod(df_A,modulo_size_edge_to_edge)
    df_AI=df_mod2.dot(A.I)
    df_AI.columns=['X','Y']
    xy_mse=pd.DataFrame([(df_AI-df_original).X.var(),(df_AI-df_original).Y.var()],index=['X','Y'],columns=[bins]).T
    mse_all_bins=pd.concat([mse_all_bins,xy_mse],axis=0)

    if not bins%5:
        fig, axes = plt.subplots(3, 2, figsize=(6, 9))
        axes = list(pd.DataFrame(axes).values.flatten())[::-1]
        i += 1
        plt.figure(i)
        plt.suptitle("number of bins: " + str(bins))

        plot(df_original,'1.original')
        plot(df_mod1,'2.after modulo')
        plot(df_quant,'3.after quantizer')
        plot(df_A,'4.after A')
        plot(df_AI,'5.after A.I')
        plot_bars(xy_mse)
    # print (pd.concat([df_AI,df_original,error],axis=1))
    # print("MSE:")
    # print(mse)
    # print(mse.loc['X','X'])
    # print(mse.loc['Y','Y'])


# i+=1
# plt.figure(i)
mse_all_bins.plot(title="MSE per bins",grid=True)
# print(mse_all_bins)
plt.show()









'''
with matrix instead of df:

import pandas as pd
import numpy as np
import pylab as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def random_data(cov,samples):
    xy=np.mat(np.random.multivariate_normal([0,0], cov, samples))
    return xy
def sign_mod(xy,modulo_size_edge_to_edge):
    xy+=modulo_size_edge_to_edge/2.0
    xy=(xy%modulo_size_edge_to_edge)-modulo_size_edge_to_edge/2.0
    return xy
def quantize(xy,modulo_size_edge_to_edge,number_of_bins):
    hlf=modulo_size_edge_to_edge/2.0
    bins = np.linspace(-hlf, hlf, number_of_bins+1)
    center = (bins[:-1] + bins[1:]) / 2  # which is also (q[:-1]+(q[1]-q[0])/2)
    bins[0] = -float("inf")  # otherwise the values outside the bins will get NaN
    bins[-1] = float("inf")
    xy=pd.DataFrame(xy,columns=['X','Y'])
    df=pd.DataFrame()
    df['X'] = pd.cut(xy.X, bins, labels=center).astype(float)
    df['Y'] = pd.cut(xy.Y, bins, labels=center).astype(float)
    # print(df)
    return np.mat(df)
def plot(data,title):
    pd.DataFrame(data,columns=['X','Y']).plot.scatter(x='X', y='Y', title=title, ax=axes.pop())
A=np.mat([[1,2],
          [-3,-4]])
# cov = [[10,9.5],[9.5,10]]
cov=A.I
modulo_size_edge_to_edge=8
bins=50
samples=1000


fig, axes = plt.subplots(3, 2, figsize=(6, 9))
axes=list(pd.DataFrame(axes).values.flatten())






xy_original=random_data(cov,samples)
plot(xy_original,'original')
# cov=df.dot(A)
xy_mod1=sign_mod(xy_original,modulo_size_edge_to_edge)
plot(xy_mod1,'after modulo')

xy_quant=quantize(xy_mod1,modulo_size_edge_to_edge,bins)
plot(xy_quant,'after quantizer')
# sns.jointplot(x="X", y="Y", data=xy_mod, kind="kde")

# xy_A=pd.DataFrame(np.mat(xy_quant)*A,columns=['X','Y'])
xy_A=xy_quant*A
plot(xy_A,'after A')

xy_mod2=sign_mod(xy_A,modulo_size_edge_to_edge)

xy_AI=xy_mod2.dot(A.I)
plot(xy_AI,'after AI')

error=xy_AI-xy_original
# print (error)
mse=error.T*(error)/samples
print(mse)
plt.show()
# print(df)

'''