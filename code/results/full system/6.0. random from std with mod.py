#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
import cufflinks
import plotly as py
from optparse import OptionParser
from threading import Thread
import pathos.multiprocessing as mp
from tqdm import tqdm

pd.set_option('expand_frame_repr', False)
print('done importing')
'''

this script is: - TODO update this...
'''

def sign_mod(xy, modulo_size_edge_to_edge):
    xy += modulo_size_edge_to_edge / 2.0
    xy = xy%modulo_size_edge_to_edge - modulo_size_edge_to_edge / 2.0
    return xy

def sign_mod_df(xy, modulo_size_edge_to_edge):
    xy = xy.copy()
    xy += modulo_size_edge_to_edge / 2.0
    xy = xy.mod(modulo_size_edge_to_edge) - modulo_size_edge_to_edge / 2.0
    # xy.columns=['X','Y']
    return xy



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-a","", dest="std_min", type="float", default=0.2, help='std minimum value')
    parser.add_option("-b","", dest="std_max", type="float", default=3.0, help='std maximum value')
    parser.add_option("-c", dest="save_to_csv", help="", default=False, action="store_true")
    parser.add_option("-n","", dest="samples", type="int", default=10,help='number of samples. dont put more than 100')
    parser.add_option("-m","", dest="modulo_size_edge_to_edge", type="float", default=10, help='modulo size edge to edge')
    parser.add_option("-s","", dest="simulations_per_std", type="float", default=1e4, help='number of simulations, for example 50. you better use 10k')
    parser.add_option("-j","", dest="number_of_stds", type="float", default=20, help='for example from 0.2 to 1.5 you try this number of std')
    (u,args)=parser.parse_args()
    if 0:
        u.number_of_stds=5
        u.simulations_per_std=20

    case='6.0_results_std_%g_%g_sims_%g_smpls_%g'%(u.std_min,u.std_max,u.simulations_per_std,u.samples)

    df = pd.DataFrame(list(np.linspace(u.std_min, u.std_max, u.number_of_stds))*int(round(u.simulations_per_std)), columns=['_std']).sort_values(by='_std').reset_index(drop=True)
    df=pd.DataFrame(df._std.map(lambda x: sign_mod(xy=np.random.normal(0,x,u.samples),modulo_size_edge_to_edge=u.modulo_size_edge_to_edge)).values.tolist(),index=df._std).reset_index(drop=False)
    print('done generating data')


    if u.save_to_csv:
        output_file_name=case+".csv"
        Thread(target=df.to_csv, args=[output_file_name]).start() # df.to_csv(output_file_name)
        # print('done saving data to "%s"'%output_file_name)
    if 0:
        df=df.sample(int(round(df.shape[0]/40))).sort_index()

    measurements_index=0
    data = df.copy().drop('_std', 1)
    measurements=pd.DataFrame()
    measurements['_std']=  df._std
    measurements['max_num']=data.T.abs().max()
    measurements['sampled_std']=data.T.std()
    measurements['sampled_kurtosis']=-data.T.kurtosis()
    measurements['0.5_mid_data']=(data.quantile([0.75],axis=1).reset_index(drop=True)-data.quantile([0.25],axis=1).reset_index(drop=True)).T # data.apply(lambda x: x.quantile([0.75]).reset_index(drop=True)-x.quantile([0.25]).reset_index(drop=True), axis=1)
    measurements['second_max']=np.sort(data.values)[:,-2] # data.apply(lambda x: x.abs().nlargest(2).min(), axis=1)
    measurements['second_max_min_range']=np.sort(data.values)[:,-2]-np.sort(data.values)[:,2] # data.apply(lambda x: x.nlargest(2).min()-x.nsmallest(2).max(), axis=1)
    measurements['range']=data.T.max()-data.T.min()
    measurements['truly_good_dist'] = measurements._std < 1.2
    print('done calculating measurements')

    data_measure = pd.concat([data, measurements], axis=1, keys=['data', 'measure'])
    if u.save_to_csv:
        Thread(target=data_measure.to_csv, args=[case+'_data_measure.csv']).start() # data_measure.to_csv('6.0 results data_measure.csv')

    false_alarm_table=pd.DataFrame()
    false_alarm_defaults = pd.DataFrame(columns=[True, False], index=[True, False]).fillna(0)
    number_of_thresholds=200
    criterias=[i for i in measurements.columns.values if i not in ['_std','truly_good_dist']]
    for criteria in tqdm(criterias):
        # print('now running on criteria %s'%criteria)
        # if criteria in :
        #     continue
        locals()[criteria]=pd.DataFrame()
        locals()[criteria]['truly_good_dist']=measurements.truly_good_dist
        for threshold in np.linspace(measurements[criteria].min(),measurements[criteria].max()+(measurements[criteria].max()-measurements[criteria].min())/number_of_thresholds/100,number_of_thresholds):
            locals()[criteria]['%s_%f'%(criteria,threshold)]=measurements[criteria]<threshold
            false_alarm = pd.crosstab(locals()[criteria].truly_good_dist, locals()[criteria]['%s_%f'%(criteria,threshold)], normalize='index')
            false_alarm = (false_alarm + false_alarm_defaults).fillna(0)
            false_alarm_table = false_alarm_table.append(dict(criteria=criteria, threshold=threshold, bad_missed=false_alarm[True][False], good_missed=false_alarm[False][True]), ignore_index=True)

    curves = false_alarm_table.pivot_table(values='good_missed', columns=['criteria'], index=['bad_missed'], aggfunc='max')
    fig = pd.DataFrame([1]).iplot(asFigure=True, xTitle='bad_missed', yTitle='good_missed', title='false alarm for %d samples'%u.samples)
    fig['data'] = [{'x': curves[col].dropna().index, 'y': curves[col].dropna().values, 'name': col} for col in curves.columns.values]
    py.offline.plot(fig, filename=case+'_curves.html')

    fig = measurements.astype(float).sample(int(round(measurements.size/100))).sort_index().iplot(asFigure=True)
    py.offline.plot(fig, filename = case+'_values.html')