#!/usr/bin/python3
import pandas as pd
import numpy as np
import pylab as plt
import cufflinks
import plotly as py
from glob import glob
pd.set_option('expand_frame_repr', False)


def remove_nans_from_plot(fig):
    for trace in range(len(fig['data'])):
        inx = np.where(fig['data'][trace]['y'] == '')
        fig['data'][trace]['x'] = np.delete(fig['data'][trace]['x'], inx)
        fig['data'][trace]['y'] = np.delete(fig['data'][trace]['y'], inx)
    return fig


print('reading csv')
if 0: # if you dont want to concatenate the results, you can read them one by one here
    path=r"C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo vs naive 2\results_del_00000000.csv.gz"
    path=r"../*_2/final_results.csv.gz"
    files=glob(path+'*.csv.gz')
    if not len(files):
        print('no files. exit')
        exit()
    df=pd.DataFrame()
    for file in files:
        print('now reading file %s'%file)
        df=pd.concat([df,pd.read_csv(file,index_col=[0])])
else:
    file=r"C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2\results_del_0.csv.gz"
    file=r"C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2\final_results.csv.gz"
    df=pd.read_csv(file, index_col=[0], nrows=100000)

df['modulo_size']=df.number_of_bins*df.quant_size
print('done reading csv')
print(df.nunique())
x_axis='modulo_size' if 1 else 'quant_size' # when doing by quant_size, you will see at higher values that the mse is the same at all methods and all number of bins

if 0:
    resolution=df.pivot_table(index=x_axis,columns='method',values=['error_per','mse'],aggfunc=[np.mean,np.median])
if 1:
    # resolution=df.pivot_table(index='quant_size',columns=['method','number_of_bins'],values=['error_per','mse'],aggfunc=[np.mean,np.median])
    resolution=df.pivot_table(index=x_axis,columns=['snr','number_of_bins','method'],values='mse',aggfunc=np.mean)
# resolution.columns=['100k']
# resolution['10k']=df.sample(int(df.shape[0]/10)).pivot_table(index='quant_size',columns='method',values='mse',aggfunc='median')
# resolution['5k']=df.sample(int(df.shape[0]/50)).pivot_table(index='quant_size',columns='method',values='mse',aggfunc='median')
if 0:
    a=resolution.idxmin().reset_index(drop=False)
    a=a.rename(columns={0:'quant_size'})
    a.pivot_table(index='number_of_bins',columns='method',values='total_modulo_size').plot()
# exit()
name='mse and error rate per number of bins'
resolution.to_excel(name+'.xlsx')
fig=resolution.figure(xTitle=x_axis, yTitle='mean mse', title='mean mse vs modulo size per method and number of bins')
fig=remove_nans_from_plot(fig)
py.offline.plot(fig,filename=name+'.html')
print('hi')


