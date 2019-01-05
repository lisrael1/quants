import numpy as np
import pandas as pd
import itertools
import pylab as plt

import sys
sys.path.append('../../')
import int_force


def __get_all_a_rows(a_max_num):
    '''
        we cannot have dependent rows, like 1,2 and 2,4, because A will be non invertible
        but we also dont want those duplication, because if you multiply this row at the numbers,
        the next row will just be like the first with additional multiply in factor, like doing [1,2] and then multiply in 2
        and also minus factor is the same. it's like mirroring the output
    :param a_max_num: the max number that you can find at the A matrix
    :return:
    '''
    comb = list(range(-a_max_num, a_max_num + 1))
    '''we dont want 0,0. we want only 0,1 not 0,-1 not 0,2 etc.'''
    comb.remove(0)
    comb = [comb, comb]
    all_rows = list(itertools.product(*comb))
    '''we removed all starting with 0, but we want 0,1. we dont want 0,2 because we already have 0,1'''
    all_rows += [(0, 1)]
    all_rows += [(1, 0)]
    all_rows = pd.DataFrame(all_rows, columns=['a', 'b'])
    '''now removing dependencies by normalizing all first numbers to 1 and removing duplicates'''
    all_rows['normal_b'] = all_rows.b / all_rows.a
    all_rows['abs_a'] = np.abs(all_rows.a)
    all_rows.sort_values(by='abs_a', inplace=True)  # sorting for taking 1,2 and not 2,4
    all_rows.drop_duplicates(subset='normal_b', keep='first', inplace=True)
    '''now just set them as list of matrix'''
    all_rows_in_list = [np.mat(i).reshape(1, 2) for i in all_rows[['a', 'b']].values]
    # print("we have %0d rows for A" % len(all_rows))
    return np.mat(all_rows[['a', 'b']].values)


def find_best_A(inputs, a_rows=None, debug=False):
    '''
        multiply inputs in rows, and taking the output std per row, returning the first best two lines
    :param a_rows: matrix
    :param inputs:
    :param debug:
    :return: the rows are standing, so [[0,1],[2,3]] is from rows 0,2 and 1,3
    '''
    if type(a_rows)==type(None):
        '''getting all A rows'''
        a_rows=int_force.A_mat.A_rows.all_A_rows(10)
    a_rows=np.mat(a_rows)
    inputs=pd.DataFrame(inputs)
    if debug:
        ab = pd.DataFrame(a_rows, columns='x,y'.split(','))
        # (np.mat(inputs.values)*a_rows.T).std(0)
        ab['sigma'] = inputs.dot(a_rows.T).std()  # if you have input of 10X2, for each a_row, you get output of 10X1
        ab=ab.sort_values('sigma').reset_index(drop=True)
        print(ab)
        A=np.mat(ab.loc[[0,1],['x','y']].values)
        inputs.set_index('X').plot(style='.', title='inputs before A')
        plt.legend('')
        inputs.dot(A.T).set_index(0).plot(style='.', title='after A')
        plt.legend('')
        plt.show()
    # best_std=m1.dot(a_rows.T).std().sort_values().head(2)
    best_std = inputs.dot(a_rows.T).std().nsmallest(2)
    best_inx = best_std.index.values
    A = a_rows[best_inx].T
    return A


__all_A_rows_var=None


def all_A_rows(a_max_num=15):
    global __all_A_rows_var
    if type(__all_A_rows_var)==type(None):
        __all_A_rows_var=__get_all_a_rows(a_max_num)
    return __all_A_rows_var


if __name__ ==  '__main__':
    print('hi')
    all_A_rows()
    all_A_rows()