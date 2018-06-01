#!/usr/bin/python3
import pandas as pd
import numpy as np
from optparse import OptionParser
import pylab as plt

pd.set_option('expand_frame_repr', False)

df=pd.read_csv('final_results.csv')
columns=[i for i in df.columns if not 'Unnamed' in i]
df=df[columns]
a=df.pivot_table(index='std_threshold',columns='method',values='mse',aggfunc='mean')
print('hi')