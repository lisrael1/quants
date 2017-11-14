import scipy.stats as st
import pylab as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def errors(u_size, threshold, n_std, samples):
    '''
        when you have inside threshold, you point N and outside you point U
    '''
    # u_error means that you have u and you dont have samples outsize the threshold
    u_error = (threshold * 2.0 / u_size) ** samples
    # n_error means that you have n and you have samples outsize the threshold
    n_to_be_outsize = (1 - st.norm(0, n_std).cdf(threshold)) * 2
    n_error = 1 - (1 - n_to_be_outsize) ** samples
    return u_error, n_error




def plot_single_U_N(modulo,threshold,sigma,samples):
    u_error,n_error=errors(modulo, threshold, sigma, samples)

    U_N_errors="%d samples\nthreshold: %.02g\nsigma: %.02g\nuniform edge: %.02g\nU dist, probability to detect as normal : %.02g\nN dist, probability to detect as uniform : %.02g"%(samples,threshold,sigma,modulo,u_error,n_error)
    fig1 = plt.figure()
    ax = plt.gca()
    x_axis = np.arange(-5 * sigma, 5 * sigma, 0.001)

    plt.plot(x_axis, st.norm.pdf(x_axis, 0, sigma),label="normal")
    ax.add_patch(patches.Rectangle((-modulo / 2.0, 0), modulo, .1, alpha=0.1,label="uniform"))
    ax.add_patch(patches.Rectangle((-threshold, 0), 2*threshold, .07, alpha=0.1,facecolor="red",label="threshold"))
    plt.legend(loc="best",ncol=1, shadow=True, title="in plot:")
    plt.text(-5 * sigma,0.2,U_N_errors)

    plt.show()

def U_N_error_vs_sigma(modulo,threshold,samples):
    sigma_samples=50
    df=pd.DataFrame({'sigma':np.linspace(0.1,1,sigma_samples)})
    df['samples']=samples
    df['modulo']=modulo
    df['threshold']=threshold
    df['errors']=df.apply(lambda x:errors(u_size=x.modulo, threshold=x.threshold, n_std=x.sigma, samples=x.samples),axis=1)
    df[['U error','N error']]=df.errors.apply(pd.Series)
    df['delta']=(df['U error']-df['N error']).apply(np.abs)
    return df


def U_N_error_vs_threshold(modulo,sigma,samples,threshold_samples=100):
    df=pd.DataFrame({'threshold':np.linspace(0,4.4,threshold_samples)})
    df['samples']=samples
    df['modulo']=modulo
    df['sigma']=sigma
    df['errors']=df.apply(lambda x:errors(u_size=x.modulo, threshold=x.threshold, n_std=x.sigma, samples=x.samples),axis=1)
    df[['U error','N error']]=df.errors.apply(pd.Series)
    df['delta']=(df['U error']-df['N error']).apply(np.abs)
    return df

def plot_U_N_error_vs_sigma(modulo,samples,sigma_samples,threshold_samples):
    out=pd.DataFrame()
    for sigma in np.linspace(0.3,1,sigma_samples):
        df=U_N_error_vs_threshold(modulo, sigma, samples,threshold_samples)
        a=df.delta.argmin()
        out=pd.concat([out,df.iloc[a].to_frame().T])
    return out



samples = 500
modulo = 8.8
sigma = 1
threshold=3.5
sigma_samples=100
threshold_samples=100
# U_N_error_vs_threshold(modulo,sigma,samples).set_index('threshold')[['U error','N error']].plot(grid=True,title="samples: %d , module: %g, sigma: %g "%(samples,modulo,sigma));plt.show()
# plot_single_U_N(modulo,threshold,sigma,samples)
# plot_U_N_error_vs_sigma(modulo,samples,sigma_samples,threshold_samples).set_index('sigma')[['U error','N error','threshold']].plot(title="samples: %d"%samples);plt.show()
# U_N_error_vs_sigma(modulo,threshold,samples).set_index('sigma')[['U error','N error']].plot(grid=True,title="samples: %d , module: %g, threshold: %g "%(samples,modulo,threshold));plt.show()

print("generating cases")
inx=pd.MultiIndex.from_product([[5,10,20,50],np.linspace(0.1,1,20),np.linspace(0,4.4,20),[8.8]],names=['samples','sigma','threshold','modulo'])
print("done generating cases")
df=pd.DataFrame(index=inx).reset_index(drop=False)
print("we have %d cases"%df.index.size)
# df.assign(**{'U_error':0,'N_error':0,'best_sigma':0,'best_threshold':0})
print("calc errors")
df[['U_error','N_error']]=df.apply(lambda x:pd.Series(errors(u_size=x.modulo, threshold=x.threshold, n_std=x.sigma, samples=x.samples)),axis=1)
print("done calc errors")
df['delta'] = (df['U_error'] - df['N_error']).apply(np.abs)
df[['best_sigma','best_threshold']]=df.apply(lambda x :pd.Series([False,False]),axis=1)
# df[['best_sigma','best_threshold']]=pd.DataFrame(columns=[True,False])
for name,group in df.groupby(['samples','sigma']):
    inx = group.delta.argmin()
    df.loc[inx, 'best_threshold'] = True
    # df.loc[group.index,'best_threshold']=True
    # break
for name,group in df.groupby(['samples','threshold']):
    inx=group.delta.argmin()
    df.loc[inx,'best_sigma']=True

print(df.sample(100))
print(df[df.best_threshold==True])

