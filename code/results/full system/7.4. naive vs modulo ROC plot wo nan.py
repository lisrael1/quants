#!/usr/bin/python3
import pandas as pd
import numpy as np
from optparse import OptionParser
import pathos.multiprocessing as mp
import cufflinks
import plotly as py
from tqdm import tqdm
import time

pd.set_option('expand_frame_repr', False)

df=pd.read_csv('del.csv',index_col=[0])
# pvt=df.pivot_table(values='mse',columns='method', index=['quant_size','number_of_bins','std_threshold'],aggfunc=[lambda x: x.count()/len(x),np.mean]).rename(columns={'<lambda>':'pass','mean':'mse'})
pvt=df.pivot_table(values='mse',columns='method', index=['quant_size','number_of_bins','std_threshold'],aggfunc=[np.mean]).rename(columns={'mean':'mse'})
reshape=pvt.stack().reset_index(drop=False).pivot_table(index=['number_of_bins'],values=['mse'],columns=['method'])
if 0:
    reshape.columns=reshape.columns.swaplevel(0,2)
    py.offline.plot(reshape[(23,'mse_from_modulo_method')].dropna().sort_index().iplot(asFigure=True))
else:
    py.offline.plot(reshape.mse.mse_from_modulo_method.dropna().sort_index().iplot(asFigure=True))
exit()
columns=[i[:-1] for i in reshape.columns]
f2=cufflinks.subplots([reshape[i].figure() for i in columns],subplot_titles=[str(i) for i in columns])
print('hi')
py.offline.plot(f2)