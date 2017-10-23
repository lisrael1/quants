import numpy as np
import pandas as pd
import pylab as plt

samples=5000
print('''get cov from A:''')
'''we want to find the cov that A will give use. A can fix the data with that cov'''
A=np.mat([[1,2],[3,4]])
d = np.mat(np.random.normal(0, 1, [samples, 2]))
expctd=(A.T.I*(A.I))

d = d * A.I
c=np.cov(d.T, bias=1)
print ("expected cov, calc from A:\n\t"+str(expctd).replace("\n","\n\t"))
print ("actual cov:\n\t"+str(c).replace("\n","\n\t"))
# exit()









print('''\n\n\nnow get A from cov:''')
cov=np.mat([[1,0.8],[0.8,1]])
xy=np.random.multivariate_normal([0,0], cov, samples)
actual_cov=np.cov(xy.T, bias=1)
print ("recovered cov:\n\t"+str(actual_cov).replace("\n","\n\t"))

def get_normalizing_matrix(cov):
    eig_vals, eig_vecs = np.linalg.eig(cov)  # vectors are vertical
    eig_vecs=np.mat(eig_vecs)
    std = np.mat(np.diag(np.sqrt(eig_vals))) # we want std, not variance
    norm=eig_vecs*(std.I) # we want to divide in the sqrt of the eigenvalues, so we have .I
    print ("fixing matrix:\n\t"+str(norm).replace("\n","\n\t"))
    return norm

norm=get_normalizing_matrix(cov)
uncor=xy*norm
cor_again=uncor*(norm.I)
# xy=xy*(cov.I)

pd.DataFrame(xy,columns=list("XY")).plot.scatter(x='X',y='Y',alpha=0.2)
for i in [0,1]:
    plt.arrow(0, 0, norm[0,i], norm[1,i], head_width=0.5, head_length=0.5,length_includes_head=True, fc='k', ec='k')
plt.show()

