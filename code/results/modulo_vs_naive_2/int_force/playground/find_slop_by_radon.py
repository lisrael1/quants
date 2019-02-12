import numpy as np
import pandas as pd
import plotly as py
import cufflinks
from sklearn.cluster import DBSCAN
import pylab as plt
from skimage.transform import radon, rescale
import warnings

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
        H = H[::-1].T
        theta = np.linspace(0., 180., max(H.shape), endpoint=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sinogram = radon(H, theta=theta, circle=True)
        # radon_angle=pd.DataFrame(sinogram, columns=np.linspace(0,180,sinogram.shape[0], endpoint=True)).stack().idxmax()[1]
        radon_angle=pd.DataFrame(sinogram, columns=np.linspace(-90, 90, sinogram.shape[0], endpoint=False)).rename_axis('angle', axis=1).rename_axis('offset', axis=0)  # angle is from the horizon, you will get from -90 to 90
        # this will tell us the angle. we can also use idxmax on the sinogram,
        # but the best is to find the angle that has the highest std
        # we cannot take the angle sum, because they all summed to big numbers
        radon_estimated_angle=radon_angle.std().idxmax()
        print('radon_angle %g'%radon_angle.stack().idxmax()[1])  # by single max number
        print('radon_angle %g'%radon_estimated_angle)  # trying to get all max numbers. you cannot do sum because all angles summed to the same values
        # this will tell us how much the image is just lines or just noise
        # below 3 is low correlation. and you have up to 3.5 to some cases that are at the middle
        print('overall sinogram std %g'%radon_angle.std().std())

        # if you want to see specific cases, you can use this:
        # if radon_angle.std().std()>3.5 or radon_angle.std().std()<3:
        # if radon_angle.std().std()>3.5:
        if 0:
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12*1.5, 4.5*1.5))
        ax1.set_title("Original")
        ax1.imshow(H, cmap=plt.cm.Greys_r)
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(-90, 90, 0, sinogram.shape[0]), aspect='auto')

        radon_angle[radon_estimated_angle].plot(ax=ax3, title='estimated angle %g'%radon_estimated_angle)

        fig.tight_layout()

        data=data.stack(0).reset_index(drop=False)
        fig=data.figure(kind='scatter', x='X', y='Y', categories='level_1', size=4, text='label')
        py.offline.plot(fig, auto_open=False)

        plt.show()
