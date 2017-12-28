#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
plt.rcParams.update({'figure.autolayout': True,'figure.figsize':(11,11)})#'figure.figsize':(8.3,11.7)
from  optparse import OptionParser

parser = OptionParser()
parser.add_option("-n","", dest="samples", type="int", default=1000,help='number of dots, for example 1000')
parser.add_option("-s","", dest="simulations", type="int", default=100,help='number of simulations, for example 50')
parser.add_option("-b","", dest="bins", type="int", default=200,help='number of bins, for example 200')
parser.add_option("-o",    dest="different_cov", help="if set, using same cov matrix for all simulations", default=False, action="store_true")
(u,args)=parser.parse_args()

def rand_cov():
    m=np.matrix(np.random.normal(0,1,[2,2]))
    m=m.T*m
    '''now changing the cov det to 1 but you can skip it'''
    m=m/np.sqrt(np.linalg.det(m))
    return np.mat(m)

def rand_A(max_det):
    done=0
    while not done:
        A=np.random.randint(-10,10,[2,2])
        if np.abs(np.linalg.det(A))<max_det and np.abs(np.linalg.det(A))>0.5:
            done=1
    return np.mat(A)

def hist_plot(df,title):
    sns.jointplot(data=df, x="X", y="Y",xlim=[-4.4,4.4],ylim=[-4.4,4.4])#,ax=axes.pop())#, kind="kde"
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title)





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



modulo_size_edge_to_edge=8.8
samples=u.samples
bins=u.bins
numbers_to_check=10
threshold=3.3

# cases="----inputs cases----\nA:\n%s\ncov:\n%s\nmodulo_size_edge_to_edge:\n%s\nsamples:\n%s"%(str(A),str(cov),str(modulo_size_edge_to_edge),str(samples))
# print (cases)
# print ("inputs:")
# print ("A:\n"+str(A))
# print ("cov:\n"+str(cov))
# print ("modulo_size_edge_to_edge:\n"+str(modulo_size_edge_to_edge))
# print ("samples:\n"+str(samples))


results=[]
for sim in range(u.simulations):
    # if not sim%8:
    #     fig, axes = plt.subplots(4, 2)#, figsize=(6, 9))
    #     axes = list(pd.DataFrame(axes).values.flatten())[::-1]
    #     plt.suptitle("number of bins: " + str(bins))
    A = rand_A(5)
    if u.different_cov or sim==0:
        cov = rand_cov()
    if sim==0:
        cov = A.T.I * (A.I)
    df_original=random_data(cov,samples)
    df_mod1=sign_mod(df_original,modulo_size_edge_to_edge)
    df_quant=quantize(df_mod1,modulo_size_edge_to_edge,bins)
    df_A=df_quant.dot(A)
    df_A.columns=['X','Y']
    df_mod2=sign_mod(df_A,modulo_size_edge_to_edge)
    df_AI=df_mod2.dot(A.I)
    df_AI.columns=['X','Y']
    # xy_mse=pd.DataFrame([(df_AI-df_original).X.var(),(df_AI-df_original).Y.var()],index=['X','Y'],columns=[bins]).T
    U=sum(pd.cut(df_mod2.head(numbers_to_check).stack().values, [-float("inf"), -threshold, threshold, float("inf")], labels=[2, 0, 1]))
    results+=[{'data':df_mod2,'var':(df_AI-df_original).var().values,'sum_vars':sum((df_AI-df_original).var()),'N_dist':not bool(U)}]
    # hist_plot(df_mod2, str(xy_mse.values))

'''now sorting by mse'''
results=np.array(results)
# inx=np.argsort([i['sum_vars'] for i in results])
inx=np.argsort([i['N_dist'] for i in results])[::-1]
results=results[inx]

for i in results:
    hist_plot(i['data'], "var = %s\nN dist by #%0d samples = %s "%(str(i['var']),numbers_to_check*2.0,i['N_dist']))

output_pdf="msv_vs_hist.pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf)
for fig in range(1, plt.figure().number): ## will open an empty extra figure :(
    pdf.savefig(fig,bbox_inches='tight',pad_inches=1)
    print ("saving fig "+str(fig))
pdf.close()
# plt.savefig("hi.pdf", format='pdf')