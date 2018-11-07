#!/usr/bin/python3
import pandas as pd
import numpy as np
from optparse import OptionParser
from tqdm import tqdm

pd.set_option('expand_frame_repr', False)

if __name__ == '__main__':
    df={'method': {0: 'mse_from_modulo_method',
  1: 'mse_from_modulo_method',
  2: 'mse_from_modulo_method',
  3: 'mse_from_modulo_method',
  4: 'mse_from_modulo_method',
  5: 'mse_from_modulo_method',
  6: 'mse_from_modulo_method',
  7: 'mse_from_modulo_method',
  8: 'mse_from_modulo_method',
  9: 'mse_from_modulo_method',
  10: 'mse_from_modulo_method',
  11: 'mse_from_modulo_method',
  12: 'mse_from_naive_method',
  13: 'mse_from_naive_method',
  14: 'mse_from_naive_method',
  15: 'mse_from_naive_method',
  16: 'mse_from_naive_method',
  17: 'mse_from_naive_method',
  18: 'mse_from_naive_method',
  19: 'mse_from_naive_method',
  20: 'mse_from_naive_method',
  21: 'mse_from_naive_method',
  22: 'mse_from_naive_method',
  23: 'mse_from_naive_method'},
 'number_of_bins': {0: 1,
  1: 3,
  2: 5,
  3: 7,
  4: 9,
  5: 11,
  6: 13,
  7: 15,
  8: 17,
  9: 19,
  10: 21,
  11: 23,
  12: 1,
  13: 3,
  14: 5,
  15: 7,
  16: 9,
  17: 11,
  18: 13,
  19: 15,
  20: 17,
  21: 19,
  22: 21,
  23: 23},
 'quant_size': {0: 0.132,
  1: 3.1020000000000003,
  2: 1.7819999999999998,
  3: 1.254,
  4: 0.99,
  5: 0.792,
  6: 0.66,
  7: 0.5940000000000001,
  8: 0.528,
  9: 0.462,
  10: 0.462,
  11: 0.396,
  12: 1.98,
  13: 1.386,
  14: 1.1880000000000002,
  15: 0.924,
  16: 0.792,
  17: 0.66,
  18: 0.5940000000000001,
  19: 0.528,
  20: 0.462,
  21: 0.462,
  22: 0.396,
  23: 0.396}}
    df = pd.DataFrame(df).iloc[6:7]
    df['std_threshold']=300
    df['number_of_sims']=0
    simulations=int(1e2)
    simulations=int(1e6)
    #df=pd.concat([df for i in range(simulations)]).reset_index(drop=True)
    c=[df]*simulations
    print('done list')
    #df=pd.concat([df for i in tqdm(range(simulations))],ignore_index=True).reset_index(drop=True)
    df=pd.concat(c,ignore_index=True).reset_index(drop=True)
    print('done generating all cases')
    #df.to_csv('simulation_cases.csv.gz',compression='gzip')
    df.to_csv('simulation_cases.csv')
