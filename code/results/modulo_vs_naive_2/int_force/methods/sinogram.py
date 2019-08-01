import sys
sys.path.append('../../')
import int_force
import pandas as pd
import numpy as np


def folded_angles(number_of_shift_per_direction):
    '''
        if we duplicate the modulo to one left, right up and down, we can get folded wave into the adjacent duplication
        so wrapped angles are 0, 45, and 90
        for double duplication for each side, we will get also 63.43, and 26.565 (by arctan(2 and 1/2)
    :param number_of_shift_per_direction:
    :return:
    '''
    if number_of_shift_per_direction==0:
        return []
    import numpy as np
    import pandas as pd
    inx = pd.MultiIndex.from_product([list(range(1, number_of_shift_per_direction + 1))] * 2, names=['x', 'y'])
    df = pd.DataFrame(index=inx).reset_index(drop=False)  # .astype(int)
    df['tan'] = df.apply(lambda row: np.rad2deg(np.arctan(row.y / row.x)), axis=1)
    return df.tan.unique().tolist() + [0, 90, -90]


def find_closest(number_of_shift_per_direction, data_after, mod_size, angle_by_std, plot):
    df = int_force.rand_data.rand_data.all_data_origin_options(data_after, mod_size, number_of_shift_per_direction=number_of_shift_per_direction, debug=plot)

    'finding closest'
    rad = np.deg2rad(-angle_by_std)
    rotation_matrix = np.matrix([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    tmp = pd.DataFrame((rotation_matrix * np.mat(df[['X', 'Y']].values).T).T)
    # tmp -= tmp.mean()
    tmp.columns = ['x_after_rotation', 'y_after_rotation']
    df = df.join(tmp)
    del tmp

    df['major_distance'] = df.y_after_rotation.abs()  # after rotation, y is the distance from the main line
    df['minor_distance'] = df.x_after_rotation.abs()  # in case he
    df['distances_angle'] = df.apply(lambda row: int_force.rand_data.find_slop.vector_to_angle(abs(row.minor_distance), abs(row.major_distance)), axis=1)
    # df['distances_angle'] = int_force.rand_data.find_slop.vector_to_angle(df.minor_distance.abs().values, df.major_distance.abs().values)
    df['distance'] = df.minor_distance / 50 + df.major_distance  # in case we have [9,0.1] and [1,0.11] we rather take the second and not the first, so we want to punish on big x distance

    # df['distance']=df.minor_distance*df.major_distance

    def group_min(group):
        # y_max=group.Y.max()
        # duplicatons_per_side=(np.sqrt(group.shape[0])-1)/2
        # smallest = group[(group.major_distance < 0.3) & (group.distances_angle < 3)]  # TODO how to set this number? for duplication of 3 times of the modulo, at folded angels we will get 7 times
        smallest = group[group.major_distance < 0.3]  # TODO how to set this number? for duplication of 3 times of the modulo, at folded angels we will get 7 times
        if smallest.empty:
            idx = group.major_distance.idxmin()
        else:  # if we have only 1 with small major_distance, we will get it here
            # idx = smallest.minor_distance.idxmin()
            idx = smallest.distance.idxmin()
        return group.loc[idx]

    tmp = df.groupby(['x_at_mod', 'y_at_mod']).apply(group_min).reset_index(drop=True)

    tmp_first_level_column = tmp.columns.to_series().replace(['x_at_mod', 'y_at_mod', 'x_center', 'y_center'], 'remove').replace(list('XY'), 'recovered').replace(['y_per_x_ratio', 'distance', 'axis_root_distance', 'closest_to_slop'], 'stat').values
    tmp.columns = [tmp_first_level_column, tmp.columns.values]

    if plot:
        red_index=pd.merge(df, df.loc[[df.sample(frac=0.01).minor_distance.idxmax()], ['x_at_mod','y_at_mod']], how='inner').drop_duplicates().index
        # red_index=pd.merge(df,df.loc[df.sample(1).index, ['x_at_mod','y_at_mod']], how='inner').index
        # red_index=pd.merge(df,df.loc[[df.loc[df.major_distance.nsmallest(5).index].minor_distance.idxmax()], ['x_at_mod','y_at_mod']], how='inner').index
        # red_index=pd.merge(df,df.loc[df[df.distance==df.distance[1:].median()].index.values, ['x_at_mod','y_at_mod']], how='inner').index
        plot_multi_modulo_before_after_rotation(angle_by_std,
                                                mod_size,
                                                df_before_rotation=df[['X', 'Y']],
                                                df_after_rotation=df[['x_after_rotation', 'y_after_rotation']].rename(columns=dict(x_after_rotation='X', y_after_rotation='Y')),
                                                red_index=red_index)
    return tmp


def plot_multi_modulo_before_after_rotation(angle_by_std, mod_size, df_before_rotation, df_after_rotation, red_index=None):
    '''

    :param angle_by_std: so you will have line that shows the sinogram estimation
    :param mod_size:
    :param df_before_rotation: df with X and Y coloumns
    :param df_after_rotation: df with X and Y coloumns
    :param red_index: you get each point x times. you can give index to color those index in black color so you can see the multiplications of this sample
    :return:
    '''
    import pylab as plt
    # checking if rotation worked
    # fig = plt.figure()
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(22, 11))
    ax1.set_title('rotating multi modulo for finding best match to angle %g' % angle_by_std)
    max_val1=df_before_rotation.abs().values.max()
    ax1.axis([-max_val1, max_val1, -max_val1, max_val1])
    ax1.set_xlim(-max_val1, max_val1)
    ax1.set_ylim(-max_val1, max_val1)
    df_before_rotation.set_index('X').Y.plot(style='.', label='original', ax=ax1, alpha=0.1)
    # df.set_index('x_after_rotation').y_after_rotation.plot(style='.', grid=True, label='after rotate to 0', ax=ax1)
    # plotting sinogram estimation angle
    ax1.plot([0, max_val1 * np.cos(np.deg2rad(angle_by_std))], [0, max_val1 * np.sin(np.deg2rad(angle_by_std))])
    # plotting modulo borders
    for i in range(-5, 5):
        if abs(i+0.5)*mod_size<max_val1:
            ax1.axvline((i+0.5)*mod_size, color='g', alpha=0.8)
            ax1.axhline((i+0.5)*mod_size, color='g', alpha=0.8)
    ax1.plot([-mod_size / 2, -mod_size / 2, mod_size / 2, mod_size / 2, -mod_size / 2],
             [-mod_size / 2, mod_size / 2, mod_size / 2, -mod_size / 2, -mod_size / 2])  # plotting the modulo frame
    ax1.grid(color='w')

    ax2.set_title('zoom in on rotated data')
    max_val2=df_after_rotation.abs().values.max()
    ax2.axis([-max_val2, max_val2, -max_val2, max_val2])
    alpha=0.1 if df_after_rotation.shape[0] > 25*500 else 0.4
    df_after_rotation.set_index('X').query('abs(Y)<0.1').Y.plot(style='.', grid=True, label='after rotate to 0', ax=ax2, c='g', alpha=alpha)
    df_after_rotation.set_index('X').query('abs(Y)>0.1').Y.plot(style='.', grid=True, label='after rotate to 0', ax=ax2, c='r', alpha=alpha/10)

    if red_index is not None:
        df_before_rotation.loc[red_index].set_index('X').Y.plot(style='x', grid=True, label='after rotate to 0', ax=ax1, c='k', alpha=1)
        df_after_rotation.loc[red_index].set_index('X').Y.plot(style='x', grid=True, label='after rotate to 0', ax=ax2, c='k', alpha=1)

    # fig.legend()


def sinogram_method(samples, number_of_bins, quant_size, snr, A_rows=None, A=None, cov=None, debug=False, plot=False):
    '''
        doesnt need to know the data covariance.
        it find the slop by sinogram, putting replica of the data next to each other,
        doing rotating by this slop, and taking the ones that are the most closest to y=0
    :param samples:
    :param number_of_bins:
    :param quant_size:
    :param snr:
    :param A_rows:
    :param A:
    :param cov:
    :param debug:
    :return:
    '''
    import pandas as pd
    pd.set_option("display.max_columns", 1000)  # don’t put … instead of multi columns
    pd.set_option('expand_frame_repr', False)  # for not wrapping columns if you have many
    pd.set_option("display.max_rows", 100)
    pd.set_option('display.max_colwidth', 1000)

    import numpy as np

    mod_size=number_of_bins*quant_size

    if type(cov)==type(None):
        cov = int_force.rand_data.rand_data.rand_cov(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    cov_angle=int_force.rand_data.find_slop.get_cov_ev(cov)[1]
    while 0:
        cov = int_force.rand_data.rand_data.rand_cov(snr=snr)
        cov_angle = int_force.rand_data.find_slop.get_cov_ev(cov)[1]
        # if not sum(abs(np.array(folded_angles(3))-abs(cov_angle))<5):
        # if not sum(abs(np.array([0, 90, -90])-abs(cov_angle))<5):
        # if abs(cov_angle)<5:
        if abs(abs(cov_angle) - 90) < 5:

            break

    data = int_force.rand_data.rand_data.random_data(cov, samples)

    '''modulo and quantization'''
    tmp = int_force.methods.methods.sign_mod(data, mod_size)
    tmp = int_force.methods.methods.to_codebook(tmp, quant_size, 0)
    tmp = int_force.methods.methods.from_codebook(tmp, quant_size, 0)
    tmp.columns = [['after'] * 2, tmp.columns.values]
    data.columns = [['before'] * 2, data.columns.values]
    data = data.join(tmp)
    del tmp

    'doing sinogram'
    hist_bins_max = 300
    sinogram_dict = int_force.methods.sinogram.calc_sinogram(data.after.X.values, data.after.Y.values, hist_bins_max=hist_bins_max, plot=plot, quant_size=quant_size)
    # if sum(abs(abs(sinogram_dict['angle_by_std'])-np.array([0, 90, -90])) < 5):
    #     number_of_shift_per_direction = 1
    # else:
    #     number_of_shift_per_direction = 3
    tmp = int_force.methods.sinogram.find_closest(number_of_shift_per_direction=3, data_after=data.after, mod_size=mod_size, angle_by_std=sinogram_dict['angle_by_std'], plot=plot)
    data = pd.merge(tmp, data, left_on=[('remove', 'x_at_mod'), ('remove', 'y_at_mod')], right_on=[('after', 'X'), ('after', 'Y')], how='right').T.sort_index().T.drop('remove', axis=1)
    del tmp
    error=data.recovered - data.before
    error.columns = [['error']*2, error.columns.values]
    data = data.join(error)
    del error
    mse = data.error.pow(2).values.mean()
    rmse=mse ** 0.5
    # if plot:
    #     plot_multi_modulo_before_after_rotation(sinogram_dict['angle_by_std'], mod_size, df_before_rotation=data.after, df_after_rotation=data.droplevel(1, axis=1)[['x_after_rotation', 'y_after_rotation']].rename(columns=dict(x_after_rotation='X', y_after_rotation='Y')))
    if (debug and rmse>(10*quant_size/np.sqrt(12))) or plot:
        import pylab as plt

        print(cov)
        # print("cov diagonal is %s" % cov.diagonal())
        print('angle by sinogram %g'%sinogram_dict['angle_by_std'])
        print('angle by ev %g'%cov_angle)
        print('mse %g'%mse)
        print('rmse %g'%rmse)
        angles_that_wraps_into_itself = folded_angles(3)  # [0, 45, 90, 26.565, 63.435] + [18.434, 33.69, 56.309, 71.565]
        if sum(abs(np.array(angles_that_wraps_into_itself) - abs(cov_angle))<1):
            print('angle too close to folded angle')
        print('big errors:')
        print(data[data.error.abs().max(axis=1)>10*quant_size/np.sqrt(12)])
        # if data[data.error.max(axis=1)>10*quant_size/np.sqrt(12)].empty:
        #     print('hi')
        print(data.loc[data.error.abs().max(axis=1).sort_values(ascending=False).index.values[:5]])

        int_force.methods.sinogram.calc_sinogram(data.after.X.values, data.after.Y.values, hist_bins_max=hist_bins_max, plot=True, quant_size=quant_size)

        # original and recovered data
        if 1:
            fig = plt.figure()
            fig.suptitle('angle %g, MSE %g' % (sinogram_dict['angle_by_std'], mse))
            data.recovered.set_index('X').Y.plot(style='.', alpha=0.4, label='recovered')
            data.after.set_index('X').Y.plot(style='.', alpha=0.4, label='after')
            data.before.set_index('X').Y.plot(style='.', alpha=0.4, label='original')
            fig.legend()
            # if mse > 1e-7 or 1:
            #     plt.show()
            # else:
            #     plt.close('all')
        else:
            import plotly as py
            import cufflinks
            fig = data[['before', 'recovered']].stack(0).reset_index(drop=False).figure(kind='scatter', x='X', y='Y', categories='level_1', size=4)
            py.offline.plot(fig)
        plt.show()
    res=dict(rmse=rmse,
             error_per=0,
             pearson=pearson,
             A=None,
             samples=samples,
             angle=sinogram_dict['angle_by_std'],
             cov_ev_angle=cov_angle,
             cov = str(cov.tolist()))
    return res


def calc_sinogram(x, y, hist_bins_max=300, quant_size=0, drop_90=False, plot=False):
    '''

    :param x: list or np list
    :param y: list or np list
    :param hist_bins_max: will try to set the minimum size for the sinogram so it will be faster, so hist_bins_max will be max value
    :param quant_size:
    :param drop_90: i had a lot of errors at this angle (and also 0) but keep this value at False
    :param plot:
    :return:
    '''
    import numpy as np
    import pandas as pd
    import warnings
    from skimage.transform import radon  # , rescale, iradon, iradon_sart, hough_line
    from scipy import signal

    max_num=max(max(x),max(y))
    if quant_size:
        bins=np.arange(-(max_num/quant_size+1),(max_num/quant_size+1))*quant_size+quant_size*0.5
    if not quant_size or len(bins)>hist_bins_max:
        bins=np.linspace(-max_num, max_num, hist_bins_max, endpoint=True)
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins]*2)  # in order to estimate the angle, we need the space to be square
    H = H[::-1].T
    if 0:  # maybe to get more resolution
        a=np.vstack([H,H])
        H=np.hstack(([a,a]))
    theta = np.linspace(0., 180., max(H.shape), endpoint=False)
    if 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sinogram = radon(H, theta=theta, circle=True)
        # sinogram=pd.DataFrame(sinogram, columns=np.linspace(0,180,sinogram.shape[0], endpoint=True)).stack().idxmax()[1]
        sinogram = pd.DataFrame(sinogram, columns=np.linspace(-90, 90, sinogram.shape[0], endpoint=False)).rename_axis('angle', axis=1).rename_axis('offset', axis=0)  # angle is from the horizon, you will get from -90 to 90
    else:  # hough gives very noisy output
        out, angles, d = hough_line(H, theta=theta)
        sinogram = pd.DataFrame(out, columns=angles, index=d).rename_axis('angle', axis=1).rename_axis('offset', axis=0)  # angle is from the horizon, you will get from -90 to 90

    if drop_90:
        sinogram[-90] = 0
    if 0:
        # this will tell us the angle. we can also use idxmax on the sinogram,
        # but the best is to find the angle that has the highest std,
        # becaues max is by single line and std is by all lines
        # we cannot take the angle sum, because they all summed to big numbers
        radon_estimated_angle = sinogram.std().idxmax()
    else:
        # max_peaking = signal.find_peaks(sinogram.std(), distance=sinogram.shape[1]//100)[0]
        max_peaking = signal.find_peaks(sinogram.std(), distance=4)[0].tolist()  # we want to find peaks so all values next to the peak will dropped
        max_peaking+=[0,sinogram.shape[0]-1]  # adding also the edges because they are not count as peaks
        max_peaking=np.array(max_peaking)
        # print(max_peaking)
        max_peaking=sinogram.std().iloc[max_peaking]
        # max_peaking=max_peaking[~max_peaking.index.isin([-90, 0, -45, 90, 45])]  # we multiply the pattern to the left right up and down so obviously we tend to get those angles
        radon_estimated_angle=max_peaking.idxmax()
    slop=np.tan(np.deg2rad(radon_estimated_angle))
    if np.isnan(radon_estimated_angle):  # you will get this if all quantized values are the same
        radon_estimated_angle=0  # ignore this - putting angle to rotate the data so it will take the original option and not all the other duplications
    sinogram_dict=dict(image=H,
                       sinogram=sinogram,
                       angle_by_std=radon_estimated_angle,
                       angle_by_idxmax=sinogram.stack().idxmax()[1],
                       overall_sinogram_std=sinogram.std().std(),
                       slop=slop,
                       y_avg=y.mean(),
                       x_avg=x.mean(),
                       actual_bin_size=len(bins))
    if plot:
        plot_sinogram(sinogram_dict['image'], sinogram_dict['sinogram'], sinogram_dict['angle_by_std'])

    if 0:
        empty_sinogram = sinogram.copy().astype(int)*0
        empty_sinogram.loc[sinogram.index.to_series().median(), radon_estimated_angle]=1
        empty_sinogram=empty_sinogram.fillna(0)
        line_image=iradon(empty_sinogram.values, circle=True)# , interpolation='linear')
        line_image.max()
        line_image.min()
        line_image.mean()
        abc=pd.DataFrame(line_image)
        # line_image=(line_image > line_image.mean()) * 255
        # line_image = iradon_sart(sinogram.values, theta=theta)
        plt.close()
        plt.imshow(line_image, cmap=plt.cm.Greys_r)
        fig=abc.figure(kind='heatmap', colorscale='Greys', title='radon inverse - real world')
        fig=empty_sinogram.figure(kind='heatmap', colorscale='Greys', title='radon inverse - real world')
        py.offline.plot(fig)

    # print('sinogram %g' % radon_estimated_angle)  # trying to get all max numbers. you cannot do sum because all angles summed to the same values
    # this will tell us how much the image is just lines or just noise
    # below 3 is low correlation. and you have up to 3.5 to some cases that are at the middle
    # print('overall sinogram std %g' % sinogram.std().std())
    return sinogram_dict


def plot_sinogram(image, sinogram, angle_by_std):
    import numpy as np
    import pylab as plt
    hist_bins_max=image.shape[0]-1  # image should be square
    picture_size_factor=1
    fig, ax = plt.subplots(2, 2, figsize=(20 * picture_size_factor, 20 * picture_size_factor))  # , subplot_kw ={'aspect': 1.5})#, sharex=False)
    ax = ax.flatten()

    ax[0].set_title("data")
    # image=image/-image.max()+1
    # sinogram=sinogram/-sinogram.max()+1
    ax[0].axis([0, hist_bins_max, 0, hist_bins_max])
    ax[0].imshow(image[::-1], cmap=plt.cm.Greys_r)
    image_middle = hist_bins_max // 2
    ax[0].plot(
        [image_middle, image_middle+20 * np.cos(np.deg2rad(angle_by_std))],
        [image_middle, image_middle+20 * np.sin(np.deg2rad(angle_by_std))],
        c='r', linewidth=4)

    ax[1].set_title("sinogram")
    y_axis_values=hist_bins_max
    y_axis_values=180  # just to square the image although real size is hist_bins_max
    ax[1].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(-90, 90, 0, y_axis_values))

    sinogram[angle_by_std].plot(ax=ax[2], title='sinogram values at angle')  # , figsize=[20, 20]

    sinogram.std().plot(ax=ax[3], title='estimated angle %g' % angle_by_std)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    # plt.show()
    print('closing picture')


def _debug_sinogram_method():
    import numpy as np
    import pandas as pd

    samples, number_of_bins, quant_size=300, 19, 1.971141
    samples, number_of_bins, quant_size=100, 101, 0.06
    cov=np.mat([[1, 0.9], [0.9, 1]])
    cov=None

    snr=100000
    results=pd.DataFrame()
    for i in range(100):
        res=sinogram_method(samples, number_of_bins, quant_size, snr, cov=cov, debug=False, plot=True)
        results=results.append(pd.Series(res), ignore_index=True)
        # print('rmse %g, angle %g' % (res['rmse'], res['angle']))
        print('rmse %g' % (res['rmse']))
    # quant_size/=10 # so we can see some errors
    print(results.describe())
    # rmse=sinogram_method(samples, number_of_bins, quant_size, snr, cov=cov, debug=True)


def compare_sinogram_and_eigen_vector(quant_size=False, hist_bins_max=300, snr=10000, samples=1000):
    import sys
    sys.path.append('../../')
    sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
    import int_force
    from sys import platform

    import numpy as np
    import pandas as pd

    # cov=np.mat([[0, 0],[0, 1]])
    # cov=np.mat([[1, 1],[1, 1.2]])
    # cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
    cov=int_force.rand_data.rand_data.rand_cov(snr=snr)
    if "win" in platform and 0:
        ang=int_force.rand_data.find_slop.get_cov_ev(cov, False)[1]
        if ang>-87 and ang<87:
            return
    data = int_force.rand_data.rand_data.random_data(cov, int(samples))
    if quant_size:
        data = int_force.methods.methods.to_codebook(data, quant_size, 0)
        data = int_force.methods.methods.from_codebook(data, quant_size, 0)

    sinogram_dict = int_force.methods.sinogram.calc_sinogram(data.X.values, data.Y.values, hist_bins_max=hist_bins_max, plot=False, quant_size=quant_size)

    res=pd.DataFrame(dict(sinogram=sinogram_dict['angle_by_std'], ev=int_force.rand_data.find_slop.get_cov_ev(cov, False)[1]), index=[1])
    res.loc[res.sinogram == -90] = 90  # ev likes positive and sinogram negative...
    res['error'] = (res.ev - res.sinogram).abs() % 45
    if "win" in platform and 1:
        # print('*' * 150)
        # print(cov)
        if res['error'].abs().values[-1] > 15:
            import pylab as plt
            print('hi')
            print(cov)
            print("cov det is %g" % np.linalg.det(cov))
            print("image size is %d" % sinogram_dict['image'].size)  # sometimes i get image at the size of 3X3 so sinogram doesnt have resolution for this. so the real problem is that the cov happen to be small
            print(res)
            int_force.methods.sinogram.calc_sinogram(data.X.values, data.Y.values, hist_bins_max=hist_bins_max, plot=True, quant_size=quant_size)
            plt.show()
            print('hi')
        # print(res[res.error.abs()>5])
        # print(res.describe())
    return res.iloc[0].round(4).to_dict()#.error.values[0]


def __debug_angle_error_per_number_of_bins_snr_and_samples():
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    tqdm.pandas()

    # a = [[10, 100, 1000, 10000], [100, 300], [100, 1000, 10000], [None]]
    # a = [[100], [100], [10000], [0.1, 0.01]]
    # inx = pd.MultiIndex.from_product(a, names=['samples', 'hist_bins_max', 'snr', 'quant_size'])
    a = dict(samples=[10, 100, 1000, 10000], hist_bins_max=[100, 300], snr=[100, 1000, 10000], quant_size=[0])
    a = dict(samples=[100], hist_bins_max=[100], snr=[10000], quant_size=[0.2, 0.1, 0.01, 0.0001])
    inx = pd.MultiIndex.from_product(a.values(), names=a.keys())

    df = pd.DataFrame(index=inx).reset_index(drop=False).sample(frac=1)
    df = pd.concat([df] * 10000, ignore_index=True)

    df = df.join(df.progress_apply(lambda row: compare_sinogram_and_eigen_vector(**row.to_dict()), axis=1).apply(pd.Series))
    df.to_csv('angle_error_by_sinogram.csv', header=None, mode='a')
    # df.to_csv('angle_error_by_sinogram.csv', mode='w')
    print(df.head())


if __name__ == '__main__':
    _debug_sinogram_method()
    # __debug_angle_error_per_number_of_bins_snr_and_samples()
