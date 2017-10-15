import numpy as np
import pandas as pd
import pylab as plt
import itertools,random

def plot_by_A(A):
    d=np.mat(np.random.normal(0,1, [50,2]))
    d=d*A.I
    df = pd.DataFrame(d, columns=['a', 'b'])
    df['A']=str(A)
    return df


n=4
a=range(-n,n)
a=[a,a,a,a]
As=[np.mat(i).reshape(2,2) for i in list(itertools.product(*a))]
# As=[np.mat([[1,-1],[1,-2]]),np.mat([[1,-1],[10,-2]]),np.mat([[-1,-1],[10,-2]]),np.mat([[1,1],[3,-2]]),np.mat([[1,1],[1,2]]),np.mat([[1,-1],[3,3]])]
As=[i for i in As if round(np.linalg.det(i))==2 and list(i.A1).count(0)<2]
# print(As)
print(len(As))
# exit()

random.shuffle(As)

df=pd.DataFrame()
for A in As[:18]:
    tmp=plot_by_A(A)
    df=pd.concat([df,tmp],axis=0)

fig, axes = plt.subplots(3, 6, figsize=(6, 9))
axes=list(pd.DataFrame(axes).values.flatten())
r=10
for name,group in df.groupby('A'):
    # group.plot(x='a', y='b', kind="scatter",grid=True, xticks=range(-r, r), sharex=True,sharey=True,yticks=range(-r, r),ax=axes.pop(), label="A=" + name.replace("\n", "\n     "))
    axes[-1].add_patch(plt.Circle((0,0), radius=2, color='r',fill=False,alpha=0.4))
    group.plot(x='a', y='b', kind="scatter",grid=True, xticks=range(-r, r),yticks=range(-r, r),ax=axes.pop(), label="A=" + name.replace("\n", "\n     "))
# those did not work - you have to give it implicitly the ax, and not just use subplots=True
# df.plot(x=['a','c'], y=['b','d'], kind="scatter",grid=True, xticks=range(-r, r), yticks=range(-r, r), label="A=" + str(A).replace("\n", "\n     "), subplots=True)
# df.plot(x=['a','c'], y=['b','d'], kind="scatter",subplots=True, sharex=False,ax=axes, legend=False)
# plt.tight_layout(pad=0, w_pad=-1, h_pad=-1)
plt.show()
