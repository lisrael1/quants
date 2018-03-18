#!/usr/bin/python3
'''
this script is:
    generating all combinations of A
    in loop:
        randomizing data det
        getting few samples from data
        for each A:
            deciding by tails of the samples if A is good
            deciding by A*cov*A if A is good
            checking errors
'''
import pandas as pd
import numpy as np
import itertools,random
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n","", dest="samples", type="int", default=2,help='number of dots X2 because you have x and y. for example 1000. you better use 5')
parser.add_option("-s","", dest="simulations", type="int", default=10,help='number of simulations, for example 50. you better use 400')
parser.add_option("-b","", dest="bins", type="int", default=200,help='number of bins, for example 200')
parser.add_option("-t","", dest="threshold", type="float", default=2.5,help='this threshold is for deciding if data is U or N, by checking if there are samples over the threshold. this defines the tail size. bigger number will result of more detecting output as N')
parser.add_option("-m","", dest="A_max_num", type="int", default=2,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10')
parser.add_option("-o",    dest="different_cov", help="if set, using same cov matrix for all simulations", default=False, action="store_true")
(u,args)=parser.parse_args()

if 0:
    u.samples=10
    u.A_max_num=8
    u.threshold=1.3

def rand_cov():
    m=np.matrix(np.random.normal(0,1,[2,2]))
    m=m.T*m
    '''now changing the cov det to 1 but you can skip it'''
    # m=m/np.sqrt(np.linalg.det(m))
    m = m / np.random.uniform(3*np.sqrt(np.linalg.det(m)), np.sqrt(np.linalg.det(m)))
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
threshold=u.threshold

# cases="----inputs cases----\nA:\n%s\ncov:\n%s\nmodulo_size_edge_to_edge:\n%s\nsamples:\n%s"%(str(A),str(cov),str(modulo_size_edge_to_edge),str(samples))
# print (cases)
# print ("inputs:")
# print ("A:\n"+str(A))
# print ("cov:\n"+str(cov))
# print ("modulo_size_edge_to_edge:\n"+str(modulo_size_edge_to_edge))
# print ("samples:\n"+str(samples))


def run_all_A_on_cov(cov,all_A):
    misdetecting_N_as_U = 0
    misdetecting_U_as_N = 0
    good_A = 0
    good_N_detection = 0

    df_original = random_data(cov, u.samples)
    df_mod1=sign_mod(df_original,modulo_size_edge_to_edge)
    df_quant=quantize(df_mod1,modulo_size_edge_to_edge,bins)
    results=[]
    for A in all_A:
        df_A=df_quant.dot(A)
        df_A.columns=['X','Y']
        df_mod2=sign_mod(df_A,modulo_size_edge_to_edge)
        df_AI=df_mod2.dot(A.I)
        df_AI.columns=['X','Y']

        output_cov=A.T*cov*A
        # xy_mse=pd.DataFrame([(df_AI-df_original).X.var(),(df_AI-df_original).Y.var()],index=['X','Y'],columns=[bins]).T
        true_good_A=(output_cov[0,0]<1.1 and output_cov[1,1]<1.1)
        good_A_by_tail=sum(pd.cut(df_mod2.head(u.samples).stack().values, [-float("inf"), -threshold, threshold, float("inf")], labels=[2, 0, 1]))==0
        good_A+=true_good_A
        if true_good_A==good_A_by_tail and true_good_A:
            good_N_detection+=1
        if true_good_A!=good_A_by_tail:
            misdetecting_N_as_U+=true_good_A
            misdetecting_U_as_N+=good_A_by_tail
            if 0:
                print(output_cov.round(10))
                print(A.round(10))
                print("true_good_A:%s, good_A_by_tail:%s"%(true_good_A,good_A_by_tail) )
                print(df_mod2.set_index('X'))
                df_mod2.set_index('X').plot(style='.')
                plt.show()
                print("*"*30)
    # print({'misdetecting_as_U':"%5d"%misdetecting_as_U,'misdetecting_as_N':"%5d"%misdetecting_as_N,'good_A':"%5d"%good_A,'sqrt_cov_det':"%3s"%str(np.sqrt(np.linalg.det(cov)).round(2)),'prsn':"%3s"%str((cov[1,0]/np.sqrt(cov[0,0]*cov[1,1])).round(2)),'cov':str(cov.round(2))})
    return {'sqrt_cov_det':np.sqrt(np.linalg.det(cov)),'prsn':cov[1,0]/np.sqrt(cov[0,0]*cov[1,1]),'good_A':good_A,'good_N_detection':100.0*good_N_detection/good_A,'misdetecting_N_as_U':misdetecting_N_as_U,'misdetecting_U_as_N':misdetecting_U_as_N,'cov':cov.round(3)}

n=u.A_max_num
a=range(-n,n+1)
a=[a,a,a,a]
all_A=[np.mat(i).reshape(2,2) for i in list(itertools.product(*a))]
# all_A=[i for i in all_A if round(np.linalg.det(i))==2 and list(i.A1).count(0)<2 and round(np.linalg.det(i))]
all_A=[i for i in all_A if list(i.A1).count(0)<2 and round(np.linalg.det(i))]
# random.shuffle(all_A)
print("we have %0d A"%len(all_A))

# all_A=[np.mat([[1,2],[-3,-4]]),np.mat([[1,2],[3,-2]])]

all=[]
for i in range(u.simulations):
    cov = rand_cov()
    # cov = all_A[0].T.I * (all_A[0].I)
    outputs=run_all_A_on_cov(cov,all_A)
    print(outputs)
    all+=[outputs]
df=pd.DataFrame(all).round(2)
df=df.reindex_axis([i for i in df.columns if i!='cov']+['cov'],axis=1)
df.to_excel("all results_n_%d_t_%g_m_%d.xlsx"%(u.samples,u.threshold,u.A_max_num))