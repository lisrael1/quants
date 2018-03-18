#!/usr/bin/python3
'''
problem with this approach:
    it's not N or U, you also have big N, and you get big error from it, but you dont find it by U threshold

this script is:
    generating all combinations of rows to A
    in loop:
        randomizing data det
        getting few samples from data
        for each row:
            giving score to that row by tails of the samples.
                i think just by worst one the biggest value
                the system flow is modulo->quantize->A->modulo
                i think we can skip the quantization...
        finding best rows to combine into A in condition that they are not dependend
        check by A*cov*A if this A is good
'''
import pandas as pd
import numpy as np
import itertools,random
from optparse import OptionParser
import pylab as plt
import seaborn as sns
import cufflinks
import plotly as py
from scipy.stats import chi2
pd.set_option('expand_frame_repr', False)


parser = OptionParser()
parser.add_option("-n","", dest="samples", type="int", default=10,help='number of dots X2 because you have x and y. for example 1000. you better use 5')
parser.add_option("-s","", dest="simulations", type="int", default=10,help='number of simulations, for example 50. you better use 400')
parser.add_option("-b","", dest="bins", type="int", default=200,help='number of bins, for example 200')
parser.add_option("-t","", dest="threshold", type="float", default=2.5,help='this threshold is for deciding if data is U or N, by checking if there are samples over the threshold. this defines the tail size. bigger number will result of more detecting output as N')
parser.add_option("-m","", dest="A_max_num", type="int", default=2,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10')
parser.add_option("-o",    dest="different_cov", help="if set, using same cov matrix for all simulations", default=False, action="store_true")
(u,args)=parser.parse_args()



def rand_cov():
    m=np.matrix(np.random.normal(0,1,[2,2]))
    m=m.T*m
    '''now changing the cov det to 1 but you can skip it'''
    scale=np.random.uniform(0.8, 3)
    # scale=0.5
    m /= np.sqrt(np.linalg.det(m))
    m/=scale
    if 0:
        print(np.linalg.det(m))
    return np.mat(m)

def hist_plot(df,title):
    # sns.jointplot(data=df, x="X", y="Y",xlim=[-4.4,4.4],ylim=[-4.4,4.4])#,ax=axes.pop())#, kind="kde"
    sns.jointplot(data=df, x="X", y="Y")
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title)





def random_data(cov,samples):
    xy=pd.DataFrame(np.random.multivariate_normal([0,0], cov, samples),columns=['X','Y'])
    return xy
def sign_mod(xy,modulo_size_edge_to_edge):
    xy=xy.copy()
    xy+=modulo_size_edge_to_edge/2.0
    xy=xy.mod(modulo_size_edge_to_edge)-modulo_size_edge_to_edge/2.0
    # xy.columns=['X','Y']
    return xy
def quantize(xy,modulo_size_edge_to_edge,number_of_bins):
    if number_of_bins>1e3:
        return xy
    hlf=modulo_size_edge_to_edge/2.0
    bins = np.linspace(-hlf, hlf, number_of_bins+1)
    center = (bins[:-1] + bins[1:]) / 2  # which is also (q[:-1]+(q[1]-q[0])/2)
    bins[0] = -float("inf")  # otherwise the values outside the bins will get NaN
    bins[-1] = float("inf")
    df=pd.DataFrame()
    df['X'] = pd.cut(xy.X, bins, labels=center).astype(float)
    df['Y'] = pd.cut(xy.Y, bins, labels=center).astype(float)
    return df

def threshold_for_detecting_not_as_U(odds, samples, modulo):
    # the odds for getting U to act like N, are (2*threshold/modulo)^samples => (odds^(1/samples))*modulo/2=threshold
    threshold=(odds**(1/samples))*modulo/2
    print('for odds of %g, with modulo %g and %d samples, threshold is %g'%(odds,modulo,samples,threshold))
    return threshold
def threshold_for_N(odds, samples, desired_std):
    # the odds for getting U to act like N, are (2*threshold/modulo)^samples => (odds^(1/samples))*modulo/2=threshold
    threshold=np.sqrt(chi2(samples-1).ppf(odds)*desired_std/(samples-1))
    print('for odds of %g, with desired std of %g and %d samples, threshold is %g'%(odds,desired_std,samples,threshold))
    return threshold

modulo_size_edge_to_edge=8.8
modulo_size_edge_to_edge=10

if 1:
    u.samples=10
    u.A_max_num=15
    u.simulations=100
    u.bins=10000
    u.threshold=1.3
    u.threshold=threshold_for_detecting_not_as_U(1/1000,u.samples*2,modulo_size_edge_to_edge)



# cases="----inputs cases----\nA:\n%s\ncov:\n%s\nmodulo_size_edge_to_edge:\n%s\nsamples:\n%s"%(str(A),str(cov),str(modulo_size_edge_to_edge),str(samples))
# print (cases)
# print ("inputs:")
# print ("A:\n"+str(A))
# print ("cov:\n"+str(cov))
# print ("modulo_size_edge_to_edge:\n"+str(modulo_size_edge_to_edge))
# print ("samples:\n"+str(samples))


def run_all_rows_on_cov(cov,all_rows,threshold,bins):
    df_original = random_data(cov, u.samples)
    df_mod1=sign_mod(df_original,modulo_size_edge_to_edge)
    df_quant=quantize(df_mod1,modulo_size_edge_to_edge,bins)
    results=pd.DataFrame(columns=['a','b','tail_max','var1','var2','var_max'])
    for row in all_rows:
        df_dot_row=df_quant.dot(row)
        # df_dot_row.columns=['X','Y']
        df_mod2=sign_mod(df_dot_row,modulo_size_edge_to_edge)
        # df_AI=df_mod2.dot(row.I)
        # df_AI.columns=['X','Y']
        A = np.mat([row.A1, row.A1]).T

        output_cov_by_single_row=A.T*cov*A
        # true_good_A=(output_cov_by_single_row[0,0]<1.1 and output_cov_by_single_row[1,1]<1.1)
        worst_one=np.abs(df_mod2).values.max()
        results=results.append(dict(a=row[0].A1.tolist()[0],b=row[1].A1.tolist()[0],tail_max=worst_one,var1=output_cov_by_single_row[0,0],var2=output_cov_by_single_row[1,1],var_max=max(output_cov_by_single_row[0,0],output_cov_by_single_row[1,1])),ignore_index=True)
    results.sort_values(by='tail_max',inplace=True)
    # now need to remove all lines that a=0 except the first one, and the ones that are not starting with 0 divide them in a and remove duplicates
    A=np.mat(results[['a','b']].head(2).values.T)
    output_cov=A.T*cov*A
    df_after_A = pd.DataFrame(np.mat(df_original) * A,columns=['X','Y'])
    df_after_second_mod = sign_mod(df_after_A,modulo_size_edge_to_edge)
    df_after_AI=pd.DataFrame(np.mat(df_after_second_mod) * (A.I),columns=['X','Y'])
    e = df_after_AI - df_original
    mse = (e ** 2).sum().sum() / e.size
    if 0:
        # print(results.head())
        # print('A checking by cov %b'% true_good_A)
        print('max var value for both rows %f'% results.head(2).var_max.max())
        print('tail max value for both rows %f'% results.head(2).tail_max.max())
        print('mse %f'% mse)
    if 0:
        print('max tail %.2f, max var %.2f, mse %.2f'%(results.head(2).tail_max.max(),results.head(2).var_max.max(),mse))
    if 0:
        # hist_plot(pd.DataFrame(df_after_second_mod, columns=['X', 'Y']), 'hi')
        hist_plot(pd.DataFrame(df_after_AI, columns=['X', 'Y']), 'hi')
        plt.show()
    return_dict={}
    return_dict['cov_det_sqrt_sqrt']=np.sqrt(np.sqrt(np.linalg.det(cov)))
    return_dict['max_std']=np.sqrt(results.head(2).var_max.max())
    return_dict['max_tail']=results.head(2).tail_max.max()
    return_dict['smse']=np.sqrt(mse)
    return_dict['max_original_std']=np.sqrt(max(cov[0,0],cov[1,1]))
    return_dict['original_abs_pearson']=np.abs(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))
    return_dict['output_abs_pearson']=np.abs(output_cov[0,1]/np.sqrt(output_cov[0,0]*output_cov[1,1]))
    return_dict['system_detection']=1 if return_dict['max_tail']<threshold else 0
    return_dict['true_detection']=1 if return_dict['max_std']<modulo_size_edge_to_edge*0.12850167052 else 0
    return_dict['error']=np.bitwise_xor(return_dict['system_detection'],return_dict['true_detection'])
    return_dict['giving_bad_A']=np.bitwise_and(return_dict['system_detection'],return_dict['error'])
    return_dict['missing_good_A']=np.bitwise_and(return_dict['true_detection'],return_dict['error'])
    # exit()
    # results.set_index('var_max').tail_max.sort_index().plot()

    # (df_mod2 > -2.5).values.sum()
        # xy_mse=pd.DataFrame([(df_AI-df_original).X.var(),(df_AI-df_original).Y.var()],index=['X','Y'],columns=[bins]).T
        # good_A_by_tail=sum(pd.cut(df_mod2.head(u.samples).stack().values, [-float("inf"), -threshold, threshold, float("inf")], labels=[2, 0, 1]))==0
    # print({'misdetecting_as_U':"%5d"%misdetecting_as_U,'misdetecting_as_N':"%5d"%misdetecting_as_N,'good_A':"%5d"%good_A,'sqrt_cov_det':"%3s"%str(np.sqrt(np.linalg.det(cov)).round(2)),'prsn':"%3s"%str((cov[1,0]/np.sqrt(cov[0,0]*cov[1,1])).round(2)),'cov':str(cov.round(2))})
    # return {'sqrt_cov_det':np.sqrt(np.linalg.det(cov)),'prsn':cov[1,0]/np.sqrt(cov[0,0]*cov[1,1]),'good_A':good_A,'good_N_detection':100.0*good_N_detection/good_A,'misdetecting_N_as_U':misdetecting_N_as_U,'misdetecting_U_as_N':misdetecting_U_as_N,'cov':cov.round(3)}
    return return_dict

comb=list(range(-u.A_max_num,u.A_max_num+1))
comb.remove(0) # we dont want 0,0. we want only 0,1 not 0,-1 not 0,2 etc.
comb=[comb,comb]
all_rows=list(itertools.product(*comb))
all_rows+=[(0,1)]
all_rows=pd.DataFrame(all_rows,columns=['a','b'])
all_rows['normal_b']=all_rows.b/all_rows.a
all_rows['abs_a']=np.abs(all_rows.a)
all_rows.sort_values(by='abs_a',inplace=True)
all_rows.drop_duplicates(subset='normal_b',keep='first',inplace=True)
all_rows=[np.mat(i).reshape(2,1) for i in all_rows[['a','b']].values]
print("we have %0d rows"%len(all_rows))

all=[]
for i in range(u.simulations):
    print('sim %d'%i)
    cov = rand_cov()
    # cov = all_rows[0].T.I * (all_rows[0].I)
    outputs=run_all_rows_on_cov(cov,all_rows,u.threshold,u.bins)
    # print(outputs)
    all+=[outputs]
df=pd.DataFrame(all)#.round(2)
n=pd.DataFrame(['data']*df.columns.size,index=df.columns.values).T
n[['error','giving_bad_A','missing_good_A','system_detection','true_detection']]='error'
df.columns=n.T.reset_index(drop=False).T.values.tolist()[::-1]
df=df.T.sort_index().T
df['error']=df.error.astype(int,inplace=True)

# n=pd.DataFrame(df.columns.values,index=df.columns.values)
# n.replace(['error','giving_bad_A','missing_good_A','system_detection','true_detection'],'error').replace(n[~n[0].str.contains('error')],'data')
print(df.round(10))
print('giving_bad_A: %d\nmissing_good_A: %d\nfinding good A: %d'%(df.error.giving_bad_A.sum(),df.error.missing_good_A.sum(),df.error.loc[(df.error.system_detection)&(df.error.true_detection)].system_detection.sum()))
if 0:
    df.set_index('max_tail').sort_index().max_var.plot()
    df.set_index('max_tail').sort_index().mse.plot()
    plt.show()
if 0:
    # fig=df.set_index('max_tail').sort_index()[['max_var','mse']].iplot(asFigure=True)
    df.columns = df.columns.droplevel(0)
    fig=df.set_index('max_tail').sort_index().iplot(asFigure=True,kind='scatter',mode='markers',xTitle='tail max value')
    py.offline.plot(fig)
# df=df.reindex_axis([i for i in df.columns if i!='cov']+['cov'],axis=1)
# df.to_excel("all results_n_%d_t_%g_m_%d.xlsx"%(u.samples,u.threshold,u.A_max_num))