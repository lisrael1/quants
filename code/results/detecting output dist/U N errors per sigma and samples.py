import scipy.stats as st
import pylab as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sympy import simplify,nsolve,Symbol,pprint,lambdify,symbols
from sympy.stats import Normal,P
from sympy.abc import x

# def eqation():
#     print("hi")
#     u_size, n_std, samples = symbols("u_size n_std samples")
#     nor = Normal(x, 0, n_std)
#     # print(find_best_threshold_and_error(8.8,1,10))
#     u_error = simplify("(x * 2.0 / u_size) ** samples")
#     n_error = simplify("1 - (1 - 2*cdf) ** samples")
#     eqtn = n_error - u_error
#     eqtn = eqtn.subs(Symbol("cdf"), P(nor > x))
#     f=lambdify((u_size,n_std,samples),eqtn,modules=['numpy', 'sympy'])
#     print(f(8.8, 0.3483, 3))
#     exit()
#     return f
# eqation()
# print(f(8.8,0.3483,3))
# exit()
# TODO try to use lambdify - no much help because we need to solve equation with x
def find_best_threshold_and_error(u_size,n_std,samples):
    '''
        u_error is when we detect uniform as normal
    '''
    # print(find_best_threshold_and_error(8.8,1,10))
    u_n_error_ratio=3
    if 1: # much faster way...:
        u_error = simplify("u_n_error_ratio*((x * 2.0 / u_size)^samples)".replace("u_n_error_ratio",str(u_n_error_ratio)))
        cdf = "(0.5 + 0.5*erf(-x/(sqrt(2)*n_std)))"
        n_error = simplify("1 - (1-2*cdf)^ samples".replace('cdf',cdf))
    else:
        nor = Normal(x, 0, n_std)
        u_error = simplify("(x * 2.0 / u_size)^samples")
        n_error = simplify("1 - (1-2*cdf)^samples")
        n_error = n_error.subs(Symbol("cdf"),P(nor>x))
    eqtn=n_error-u_error
    # pprint(eqtn)
    subs=[(Symbol("samples"),samples),(Symbol("n_std"),n_std),(Symbol("u_size"),u_size)]
    eqtn=eqtn.subs(subs)
    threshold=nsolve(eqtn,x,(n_std,u_size/2.0),simplify=False,rational=False , solution_dict=False, verify=False)
    error=u_error.subs(x,threshold).subs(subs).evalf()
    # print("u size: %g, n std: %6.04g, samples: %2d, threshold: %6.04g, error: %6.04g"%(u_size,n_std,samples,threshold,error))
    return float(threshold),float(error)

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






samples = 5
sigma = 1
modulo = 8.8
sigma_samples=50

print("generating cases")
inx=pd.MultiIndex.from_product([[3,5,10,20,50,70],np.linspace(0.1,1,sigma_samples),[modulo]],names=['samples','sigma','modulo'])
print("done generating cases")
df=pd.DataFrame(index=inx).reset_index(drop=False)
print("we have %d cases"%df.index.size)
print("calc errors")
df[['threshold','error']]=df.apply(lambda x:pd.Series(find_best_threshold_and_error(u_size=x.modulo, n_std=x.sigma, samples=x.samples)),axis=1)
print("done calc errors")
pt=df.pivot_table(index='sigma',columns='samples',values='error')
print(pt)
pt.plot(title="probability to get wrong per sigma at different number of samples",grid=True)
plt.show()

