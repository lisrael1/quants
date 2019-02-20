import numpy as np
import pandas as pd
# import plotly as py
# import cufflinks
import sys
sys.path.append('../../')
sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
import int_force
import pylab as plt
import plotly as py
import cufflinks

pd.set_option("display.max_columns",1000) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",10)
pd.set_option('display.max_colwidth',1000)
debug=True

if __name__ == '__main__':
    for i in range(100):
        print('*'*20+'iteration number %d'%i+'*'*20)
        # from int_force.rand_data import rand_data

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        samples=1000
        number_of_bins=17
        mod_size=1.8
        quant_size=mod_size/number_of_bins
        snr=1000

        cov=int_force.rand_data.rand_data.rand_cov(snr=snr)
        data=int_force.rand_data.rand_data.random_data(cov, samples)

        '''modulo and quantization'''
        tmp=int_force.methods.methods.sign_mod(data, mod_size)
        tmp = int_force.methods.methods.to_codebook(tmp, quant_size, 0)
        tmp = int_force.methods.methods.from_codebook(tmp, quant_size, 0)
        tmp.columns=[['after']*2,tmp.columns.values]
        data.columns=[['before']*2,data.columns.values]
        data=data.join(tmp)
        del tmp

        print('doing sinogram')
        hist_bins=300
        sinogram_dict=int_force.methods.ml_modulo.calc_sinogram(data.after.X.values, data.after.Y.values, bins=hist_bins)
        df = int_force.rand_data.rand_data.all_data_origin_options(data.after, mod_size, number_of_shift_per_direction=2, debug=debug)

        print('finding closest')
        rad=np.deg2rad(-sinogram_dict['angle_by_std'])
        rotation_matrix=np.matrix([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
        tmp=pd.DataFrame((rotation_matrix*np.mat(df[['X','Y']].values).T).T)
        tmp -= tmp.mean()
        tmp.columns = ['x_after_rotation', 'y_after_rotation']
        df=df.join(tmp)
        del tmp

        df['major_distance']=df.y_after_rotation.abs()  # after rotation, y is the distance from the main line
        df['minor_distance']=df.x_after_rotation.abs()  # in case he

        df['axis_root_distance']=np.hypot(df.X.values,df.Y.values)

        idx = df.groupby(['x_at_mod', 'y_at_mod']).major_distance.idxmin()
        tmp = df.loc[idx]
        tmp_first_level=tmp.columns.to_series().replace(['x_at_mod','y_at_mod','x_center','y_center'],'remove').replace(list('XY'),'recovered').replace(['y_per_x_ratio','distance','axis_root_distance','closest_to_slop'],'stat').values
        tmp.columns = [tmp_first_level, tmp.columns.values]
        data=pd.merge(tmp, data, left_on=[('remove', 'x_at_mod'),('remove', 'y_at_mod')], right_on=[('after', 'X'),('after', 'Y')], how='inner').T.sort_index().T.drop('remove', axis=1)
        del tmp
        mse=(data.recovered-data.before).pow(2).values.mean()
        if debug:  # checking if rotation worked
            fig=plt.figure()
            fig.suptitle('rotating multi modulo for finding best match to angle %g'%sinogram_dict['angle_by_std'])
            df.set_index('X').Y.plot(style='.', grid=True, label='original')
            df.set_index('x_after_rotation').y_after_rotation.plot(style='.', grid=True, label='after rotate to 0')
            plt.plot([0,5],[0,5*np.tan(np.deg2rad(sinogram_dict['angle_by_std']))])
            plt.plot([-mod_size/2,-mod_size/2, mod_size/2, mod_size/2, -mod_size/2],[-mod_size/2,mod_size/2, mod_size/2, -mod_size/2, -mod_size/2])  # plotting the modulo frame
            fig.legend()
        if debug:
            print(data)
            print('sinogram data')
            fig, ax = plt.subplots(1, 4, figsize=(12 * 2.1, 4.5 * 2.1))  # , subplot_kw ={'aspect': 1.5})#, sharex=False)
            ax[0].set_title("data")
            ax[0].imshow(sinogram_dict['image'], cmap=plt.cm.Greys_r)
            ax[0].plot([hist_bins//2-hist_bins*sinogram_dict['x_avg'], hist_bins//2+100-hist_bins*sinogram_dict['x_avg']], [hist_bins//2-hist_bins*sinogram_dict['y_avg'], hist_bins//2-hist_bins*sinogram_dict['y_avg']-100 * sinogram_dict['slop']])

            ax[1].set_title("sinogram")
            ax[1].imshow(sinogram_dict['sinogram'], cmap=plt.cm.Greys_r)

            sinogram_dict['sinogram'][sinogram_dict['angle_by_std']].plot(ax=ax[2], title='sinogram values at angle')  # , figsize=[20, 20]

            sinogram_dict['sinogram'].std().plot(ax=ax[3], title='estimated angle %g' % sinogram_dict['angle_by_std'])

            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            print('original and recovered data')
            if 1:
                fig=plt.figure()
                fig.suptitle('angle %g, MSE %g'%(sinogram_dict['angle_by_std'], mse))
                data.recovered.set_index('X').Y.plot(style='.', alpha=0.1)
                data.before.set_index('X').Y.plot(style='.', alpha=0.1)
                if mse>1e-7 or 1:
                    plt.show()
                else:
                    plt.close('all')
            else:
                fig=data[['before','recovered']].stack(0).reset_index(drop=False).figure(kind='scatter', x='X', y='Y', categories='level_1', size=4)
                py.offline.plot(fig)

        print('done')

