#!/usr/bin/python3
import pandas as pd
import numpy as np
from optparse import OptionParser

pd.set_option('expand_frame_repr', False)

if __name__ == '__main__':
    if 1:
        parser = OptionParser()
        parser.add_option("-n", "", dest="samples", type="int", default=100, help='number of dots X2 because you have x and y. for example 1000. you better use 5')
        # parser.add_option("-s", "", dest="simulations", type="int", default=20, help='number of simulations, for example 50. you better use 400') # about 8360 sims per unit here
        parser.add_option("-s", "", dest="simulations", type="int", default=20000, help='number of simulations, for example 50. you better use 400') # about 8360 sims per unit here
        # parser.add_option("-b", "", dest="number_of_bins_range", type="str", default='[3,25]', help='number of bins, for example [3,25]')
        parser.add_option("-b", "", dest="number_of_bins_range", type="str", default='[23,25]', help='number of bins, for example [3,25]')
        # parser.add_option("-q", "", dest="quant_size_range", type="str", default='[0,2]', help='number of bins, for example [0,2]')
        parser.add_option("-q", "", dest="quant_size_range", type="str", default='[0.39,0.4]', help='number of bins, for example [0,2]')
        number_of_quant_size=1#20
        parser.add_option("-t", "", dest="std_threshold_range", type="str", default='[0.6,3]', help='number of bins, for example [0,2]')
        number_of_thresholds=20
        parser.add_option("-m", "", dest="A_max_num", type="int", default=15,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10')
        parser.add_option("-p", dest="run_serial", action='store_false', help="dont run parallel [default: %default]",default=True)
        (u, args) = parser.parse_args()


        '''420,000 rows took me 1645 sec'''
        quant_size=np.linspace(eval(u.quant_size_range)[0],eval(u.quant_size_range)[1],number_of_quant_size+1)[1:]
        std_threshold=np.linspace(eval(u.std_threshold_range)[0],eval(u.std_threshold_range)[1],number_of_thresholds)
        number_of_bins=range(eval(u.number_of_bins_range)[0],eval(u.number_of_bins_range)[1],2)
        number_of_sims=range(u.simulations)
        method=['mse_from_basic_method','mse_from_modulo_method','mse_from_naive_method']
        method=['mse_from_naive_method','mse_from_modulo_method']
        inx = pd.MultiIndex.from_product([quant_size,number_of_bins,std_threshold,method,number_of_sims], names=['quant_size', 'number_of_bins', 'std_threshold', 'method', 'number_of_sims'])
        print('generated %d simulations'%np.prod([len(i) for i in [quant_size,number_of_bins,std_threshold,method,number_of_sims]]))
        print('generated %d simulations'%inx.shape[0])
        df = pd.DataFrame(index=inx).reset_index(drop=False)
        print('done generating all cases')
        df.to_csv('simulation_cases.csv')

