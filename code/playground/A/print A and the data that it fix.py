import numpy as np
import pandas as pd
import pylab as plt
import itertools,random

'''
A composed from vertical vectors, so [[-1,-4],[1,2]] will be vectors of [-1,1] and [-4,2]
'''

def data_that_match_A(A):
    d=np.mat(np.random.normal(0,1, [50,2]))
    d=d*A.I
    df = pd.DataFrame(d, columns=['a', 'b'])
    df['A']=str(A)
    return df

def get_data_angle(A):
    cov=A.I.T*A.I
    eig_vals, eig_vecs = np.linalg.eig(cov)
    right_sort = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[right_sort]
    eig_vecs = eig_vecs[:, right_sort]
    angle=np.degrees(np.arctan(eig_vecs[0,0]/eig_vecs[1,0]))
    return eig_vals, eig_vecs,angle


n=8
a=range(-n,n+1)
a=[a,a,a,a]
As=[np.mat(i).reshape(2,2) for i in list(itertools.product(*a))]
# As=[i for i in As if round(np.linalg.det(i))<4 and list(i.A1).count(0)<2 and round(np.linalg.det(i))]
As=[i for i in As if round(np.linalg.det(i))==2 and list(i.A1).count(0)<2 and round(np.linalg.det(i))]
print(len(As))

random.shuffle(As)

df=pd.DataFrame()
for A in As[:18]:
    tmp=data_that_match_A(A)
    df=pd.concat([df,tmp],axis=0)

fig, axes = plt.subplots(3, 6, figsize=(6, 9))
axes=list(pd.DataFrame(axes).values.flatten())
r=10
for name,group in df.groupby('A'):
    axes[-1].add_patch(plt.Circle((0,0), radius=2, color='r',fill=False,alpha=0.4))
    # group.plot(x='a', y='b', kind="scatter",grid=True, xticks=range(-r, r),yticks=range(-r, r),ax=axes.pop(), label="A=" + name.replace("\n", "\n     "))# , sharex=True,sharey=True
    std,eig_vecs,angle=get_data_angle(np.mat(name).reshape((2, 2)))
    for i in [0,1]:
        axes[-1].arrow(0, 0, eig_vecs[0, i] * std[i], eig_vecs[1, i] * std[i], head_width=0.3, head_length=0.3, length_includes_head=False, fc='k', ec='k')
    group.plot(x='a', y='b', kind="scatter",grid=True, xticks=range(-r, r),yticks=range(-r, r),ax=axes.pop(), label="A=" + str(name).replace("\n", "\n     "))# , sharex=True,sharey=True


# plt.tight_layout(pad=0, w_pad=-1, h_pad=-1)
plt.figure()
for A in As:
    std,eig_vecs,angle=get_data_angle(A)
    for i in [0]:
        plt.arrow(0, 0, eig_vecs[0, i] * std[i], eig_vecs[1, i] * std[i], head_width=0.3, head_length=0.3, length_includes_head=False, fc='k', ec='k')


plt.show()
