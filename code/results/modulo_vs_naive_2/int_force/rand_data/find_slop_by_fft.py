print('i dont know how to deduce the angle from the 2D fft output...')
import numpy as np
import pandas as pd
import plotly as py
import cufflinks
from sklearn.cluster import DBSCAN
import pylab as plt
from skimage.transform import radon, rescale
import warnings


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
    main_vect=eig_vecs[0].tolist()[0]
    angle1 = np.angle(complex(main_vect[1], main_vect[0]), deg=True)
    angle2=np.degrees(np.arctan(main_vect[0]/main_vect[1]))
    print(angle1)
    print(angle2)
    return angle1, angle2


if __name__ == '__main__':
    for i in range(100):
        print('*'*150)
        import sys
        sys.path.append('../../')
        sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
        import int_force
        # from int_force.rand_data import rand_data

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        cov=int_force.rand_data.rand_data.rand_cov(snr=10000)
        # print(cov)

        data=int_force.rand_data.rand_data.random_data(cov, 1000)
        samples=1000
        quant_size=0.1
        number_of_bins=17
        mod_size=2.5
        tmp=int_force.methods.methods.sign_mod(data, mod_size)
        tmp.columns=[['after']*2,tmp.columns.values]
        data.columns=[['before']*2,data.columns.values]
        data=data.join(tmp)
        clustering = DBSCAN(eps=0.1).fit(data.after.values)
        data['after','label']=clustering.labels_
        data.head()

        H, xedges, yedges = np.histogram2d(data.after.X, data.after.Y, bins=100)
        H=H.T
        plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # fft=np.fft.fft2(H)
        fft=pd.DataFrame(np.fft.fft2(H)).stack().to_frame('fft')
        fft['real_fft']=np.real(fft.fft)
        fft['abs_fft']=np.abs(fft.fft)
        fft['angle_rad_fft']=np.angle(fft.fft)
        fft['angle_deg_fft']=np.angle(fft.fft, deg=True)
        fft.sort_values(by='abs_fft', inplace=True, ascending=False)
        fft.head()
        plt.imshow(fft.abs_fft.unstack(), interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.imshow(fft.real_fft.unstack(), interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.show()


        data=data.stack(0).reset_index(drop=False)
        fig=data.figure(kind='scatter', x='X', y='Y', categories='level_1', size=4, text='label')
        py.offline.plot(fig, auto_open=False)

        plt.show()
