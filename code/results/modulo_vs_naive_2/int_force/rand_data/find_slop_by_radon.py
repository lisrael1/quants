import sys
sys.path.append('../../')
sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
import int_force

import numpy as np
import pandas as pd
import plotly as py
import cufflinks
from sklearn.cluster import DBSCAN
import pylab as plt

import itertools

# TODO
# how many samples do we need? 20? 50? 100
# ways to un modulo:
#       plot line and do on it modulo and see each line where it came from
#       you have the slop so calculate where the next line should start
#       on the sinogram mark the places at the right angle and make them thick, so they will cover all dots at the sampled data
#       take the parts from the modulo image, and move each line until it find the right place. you have 9 option, and you should not repeat place
# we will try:
#       just put the same modulo pattern at all big picture and remove the un relevant places
#


def main_eigenvector_angle(cov):
    '''
        giving wrong angle!
    :param cov:
    :return:
    '''
    e = np.linalg.eig(cov)  # vectors are vertical
    right_sort = np.argsort(np.abs(e[0]))[::-1]  # they are not sorted!!!
    eig_vals = e[0][right_sort]
    eig_vecs = e[1][:, right_sort]
    main_vect = eig_vecs[:, 0].T.tolist()[0]
    angle1 = np.angle(complex(main_vect[1], main_vect[0]), deg=True)
    angle2 = np.degrees(np.arctan(main_vect[0] / main_vect[1]))
    print(angle1)
    print(angle2)
    print(90 - angle2)
    return angle1, angle2


if __name__ == '__main__':
    for i in range(10):
        print('*'*150)
        # from int_force.rand_data import rand_data

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        cov=int_force.rand_data.rand_data.rand_cov(snr=10000)

        # main_eigenvector_angle(cov)

        samples=50
        data=int_force.rand_data.rand_data.random_data(cov, samples)
        quant_size=0.1
        number_of_bins=17
        mod_size=1.8
        tmp=int_force.methods.methods.sign_mod(data, mod_size)
        tmp.columns=[['after']*2,tmp.columns.values]
        data.columns=[['before']*2,data.columns.values]
        data=data.join(tmp)
        # clustering = DBSCAN(eps=0.1).fit(data.after.values)
        # data['after','label']=clustering.labels_
        # data.head()

        a_lot_of_data = int_force.rand_data.rand_data.random_data(cov, 10000)
        a_lot_of_data = int_force.methods.methods.sign_mod(a_lot_of_data, mod_size)

        # if you want to see specific cases, you can use this:
        # if sinogram_by_multi_samples.std().std()>3.5 or sinogram_by_multi_samples.std().std()<3:
        # if sinogram_by_multi_samples.std().std()>3.5:
        if 0:
            continue

        datas=[]
        datas+=[data.after]
        datas+=[a_lot_of_data]
        datas+=[int_force.rand_data.rand_data.all_data_origin_options(data.after, mod_size, 5)[list('XY')]]
        datas+=[int_force.rand_data.rand_data.all_data_origin_options(a_lot_of_data, mod_size, 5)[list('XY')]]

        fig, ax = plt.subplots(len(datas), 4, figsize=(12*2.1, 4.5*2.1))#, subplot_kw ={'aspect': 1.5})#, sharex=False)
        # fig, ax = plt.subplots(len(datas), 4)
        fig.suptitle("finding image or multi gaussian rotation")

        for row in range(ax.shape[0]):
            sinogram_dict=int_force.methods.ml_modulo.calc_sinogram(datas[row].X.values, datas[row].Y.values, bins=600)
            ax[row, 0].set_title("data")
            ax[row, 0].imshow(sinogram_dict['image'], cmap=plt.cm.Greys_r)
            # ax[row, 0].imshow(sinogram_dict['line'],  cmap=plt.cm.Greys_r)

            ax[row, 1].set_title("sinogram")
            ax[row, 1].imshow(sinogram_dict['sinogram'], cmap=plt.cm.Greys_r)

            sinogram_dict['sinogram'][sinogram_dict['angle_by_std']].plot(ax=ax[row, 2], title='estimated angle %g' % sinogram_dict['angle_by_std'])  # , figsize=[20, 20]

            sinogram_dict['sinogram'].std().plot(ax=ax[row, 3], title='angles std')

        # fig.tight_layout(pad=0,w_pad=1,h_pad=1)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)  # for leaving space for the overall title. you can also do fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if 0:
            data=data.stack(0).reset_index(drop=False)
            fig=data.figure(kind='scatter', x='X', y='Y', categories='level_1', size=4, text='label')
            py.offline.plot(fig, auto_open=False)

        plt.show()
