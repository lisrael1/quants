#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib
from sys import platform
matplotlib.use('Agg')
import pylab as plt
# import seaborn as sns
import matplotlib.backends.backend_pdf
plt.rcParams.update({'figure.autolayout': True,'figure.figsize':(11,11)})
from  optparse import OptionParser

parser = OptionParser()
parser.add_option("-a","", dest="A", type="string",default='[[1,2],[-3,-4]]',help='A matrix, for example "[[1,2],[-3,-4]]"')
parser.add_option("-c","", dest="cov", type="string", default='no',help='cov matrix, for example "[[1,2],[-3,-4]]" (by default its A.T.I*(A.I)')
parser.add_option("-m","", dest="modulo_size_edge_to_edge", type="float", default=8,help='for example from -4 to 4 it will be 8')
parser.add_option("-s","", dest="samples", type="float", default=1000,help='for example 1000')
(u,args)=parser.parse_args()


# A=np.mat([[1,2],[-3,-4]])
# print ("this A can fix the cov of \n"+str(A.T.I*(A.I)))
# cov=A.T.I*(A.I)
# print (cov)
# cov = [[10,9.5],[9.5,10]]
# modulo_size_edge_to_edge=8
# samples=1000


A=np.mat(eval(u.A))
if u.cov=="no":
    cov=A.T.I*(A.I)
else:
    cov=np.mat(eval(u.cov))
modulo_size_edge_to_edge=u.modulo_size_edge_to_edge
samples=int(round(u.samples))

cases="----inputs cases----\nA:\n%s\ncov:\n%s\nmodulo_size_edge_to_edge:\n%s\nsamples:\n%s"%(str(A),str(cov),str(modulo_size_edge_to_edge),str(samples))
print (cases)
# print ("inputs:")
# print ("A:\n"+str(A))
# print ("cov:\n"+str(cov))
# print ("modulo_size_edge_to_edge:\n"+str(modulo_size_edge_to_edge))
# print ("samples:\n"+str(samples))
plt.figure()
plt.figtext(x=0.3,y=0.5,s=cases, fontsize=30, multialignment='left', color='#000066', wrap=True)


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
    pd.DataFrame(data,columns=['X','Y']).plot.scatter(x='X', y='Y', title=title, ax=axes.pop(),alpha=0.2,grid=True,figsize=(11,11), rasterized=True)
    # pd.DataFrame(data,columns=['X','Y']).plot.hexbin(x='X', y='Y', title=title,gridsize=10, ax=axes.pop())
    # sns.jointplot(x="X", y="Y", data=df_mod, kind="kde")
def plot_bars(df):
    # pd.plotting.table(axes.pop(),mse, loc='center')
    # sns.heatmap(xy_mse, annot=True,ax=axes.pop())
    df=xy_mse.copy().round(2)
    ax=df.plot.bar(ax=axes.pop(), title="6.MSE",alpha=0.6, rasterized=True)
    ax.annotate(float(df.X), xy=(0, df.X),horizontalalignment='right')
    ax.annotate(float(df.Y), xy=(0, df.Y),horizontalalignment='left')





mse_all_bins=pd.DataFrame()
i=1
for bins in list(range(3,30))+[200]:
    print("running now on "+str(bins)+" bins")
    df_original=random_data(cov,samples)
    df_mod1=sign_mod(df_original,modulo_size_edge_to_edge)
    df_quant=quantize(df_mod1,modulo_size_edge_to_edge,bins)
    df_A=df_quant.dot(A)
    df_A.columns=['X','Y']
    df_mod2=sign_mod(df_A,modulo_size_edge_to_edge)
    df_AI=df_mod2.dot(A.I)
    df_AI.columns=['X','Y']
    xy_mse=pd.DataFrame([(df_AI-df_original).X.var(),(df_AI-df_original).Y.var()],index=['X','Y'],columns=[bins]).T
    if bins!=200:
        mse_all_bins=pd.concat([mse_all_bins,xy_mse],axis=0)
    else:
        std_200=pd.concat([df_original.std(),df_AI.std()],axis=1).round(2)
        std_200.columns=['df_original','df_AI']


    if not bins%5:
        fig, axes = plt.subplots(3, 2)#, figsize=(6, 9))
        axes = list(pd.DataFrame(axes).values.flatten())[::-1]
        i += 1
        plt.figure(i)#,figsize=(117,83))
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

plt.figure()
plt.figtext(x=0.3,y=0.5,s="std:\n"+str(std_200), fontsize=30, multialignment='left', color='#000066', wrap=True)

# i+=1
# plt.figure(i)
mse_all_bins.plot(title="MSE per bins",grid=True, rasterized=True)
# print(mse_all_bins)
# plt.show()
pd.DataFrame(random_data(cov,samples), columns=['X', 'Y']).plot.scatter(x='X', y='Y', title='data and A', alpha=0.2, grid=True, rasterized=True)
for i in [0,1]:
    plt.arrow(0, 0, A[0,i], A[1,i], head_width=0.5, head_length=0.5,length_includes_head=True, fc='k', ec='k')
if "win" in platform:
    output_pdf="output.pdf"
else:
    output_pdf="/tmp/output.pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf)
for fig in range(1, plt.figure().number): ## will open an empty extra figure :(
    pdf.savefig(fig,bbox_inches='tight',pad_inches=1)
    print ("saving fig "+str(fig))
pdf.close()
# plt.savefig("hi.pdf", format='pdf')