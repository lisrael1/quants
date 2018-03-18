import pandas as pd,numpy as np
from scipy.stats import chi,chi2
import plotly as py
import cufflinks

k=20
res=pd.DataFrame()
for std in [1,1.5,2,2.5,3]:
    df=pd.DataFrame(index=np.linspace(0, 28, 1000))
    df['chi_%03d_std_%g'%(k,std)]=chi(k).pdf(df.index.values)
    '''now we fix x axis:'''
    df.index=std*df.index.values/np.sqrt(k) # k-1 for sampled std with fixing bias and k for sampled std as is
    res=pd.concat([res,df[df>1e-3].dropna()],axis=0)
fig=res.iplot(asFigure=True)
py.offline.plot(fig)