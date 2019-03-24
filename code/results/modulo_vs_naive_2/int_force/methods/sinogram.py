import sys
sys.path.append('../../')
import int_force


def sinogram_method(samples, number_of_bins, quant_size, snr, A_rows=None, A=None, cov=None, debug=False):
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
    data = int_force.rand_data.rand_data.random_data(cov, samples)

    '''modulo and quantization'''
    tmp = int_force.methods.methods.sign_mod(data, mod_size)
    tmp = int_force.methods.methods.to_codebook(tmp, quant_size, 0)
    tmp = int_force.methods.methods.from_codebook(tmp, quant_size, 0)
    tmp.columns = [['after'] * 2, tmp.columns.values]
    data.columns = [['before'] * 2, data.columns.values]
    data = data.join(tmp)
    del tmp
    # high_resolution_sim=True
    # if snr > 1e6 and number_of_bins>1000:
    #     high_resolution_sim=True
    # else:
    #     high_resolution_sim=False

    'doing sinogram'
    number_of_shift_per_direction=2  # TODO i think we can lower this to 1. the covariance doesnt have 2 cyclic loop
    hist_bins = 300
    # hist_bins = 600  # it's better than 300, although slower
    # angles_that_wraps_into_itself=[0, 45, 90, 26.565, 63.435]  # if we have data at 0|45|90 degrees, we will take the data as is without doing un modulo
    # angle_close_to_wrap=1  # TODO angle_close_to_wrap here depend on the snr. if we have big snr, we can do little number, and if low, we need bigger than angle_close_to_wrap
    # if number_of_shift_per_direction==2:
    #     angles_that_wraps_into_itself+=[18.434, 33.69, 56.309, 71.565]  # arctan(3/2 or 1/3 or 2/3 or 3)

    # if high_resolution_sim:  # at high snr, the folding modulo is less likely to happen, so we can ignore this
    #     # angles_that_wraps_into_itself=[]
    #     angle_close_to_wrap=0.5
    #     # hist_bins = 600

    # print('setting the bins to 300')

    sinogram_dict = int_force.methods.sinogram.calc_sinogram(data.after.X.values, data.after.Y.values, hist_bins=hist_bins, plot=debug)
    # angles_that_wraps_into_itself=[0, 45, 90]  # if we have data at 0|45|90 degrees, we will take the data as is without doing un modulo
    df = int_force.rand_data.rand_data.all_data_origin_options(data.after, mod_size, number_of_shift_per_direction=number_of_shift_per_direction, debug=debug)

    cov_angle=int_force.rand_data.find_slop.get_cov_ev(cov)[1]
    'finding closest'
    rad = np.deg2rad(-sinogram_dict['angle_by_std'])
    rotation_matrix = np.matrix([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    tmp = pd.DataFrame((rotation_matrix * np.mat(df[['X', 'Y']].values).T).T)
    # tmp -= tmp.mean()
    tmp.columns = ['x_after_rotation', 'y_after_rotation']
    df = df.join(tmp)
    del tmp

    df['major_distance'] = df.y_after_rotation.abs()  # after rotation, y is the distance from the main line
    df['minor_distance'] = df.x_after_rotation.abs()  # in case he

    def group_min(group):
        smallest = group[group.major_distance < 0.1]
        if smallest.empty:
            idx = group.major_distance.idxmin()
        else:
            idx = smallest.minor_distance.idxmin()
        return group.loc[idx]
    tmp = df.groupby(['x_at_mod', 'y_at_mod']).apply(group_min).reset_index(drop=True)

    # # we have 2 ways - or to check if we have angle with problems, or to check if we have multiple options that are close to the x axis and then take the closest to y axis
    # df['axis_root_distance'] = np.hypot(df.X.values, df.Y.values)
    # if high_resolution_sim:
    #     def group_min(group):
    #         # smallest = group.nsmallest(10, 'major_distance')
    #         # smallest=smallest[smallest.major_distance<3*smallest.major_distance.iloc[0]]
    #         # if smallest.major_distance.ptp() > 3 * quant_size:
    #         # if smallest.shape[0]==1:
    #         #     idx = smallest.major_distance.idxmin()
    #         # else:
    #         #     idx = smallest.minor_distance.idxmin()
    #         # smallest=group[group.major_distance<3*group.major_distance.min()]
    #         # smallest=group[group.major_distance<10*quant_size]
    #         smallest=group[group.major_distance<0.1]
    #         if smallest.empty:
    #             idx=group.major_distance.idxmin()
    #         else:
    #             idx = smallest.minor_distance.idxmin()
    #         return group.loc[idx]
    #     # tmp = df.drop_duplicates(['x_center', 'y_center'], keep='first').groupby(['x_at_mod', 'y_at_mod']).apply(group_min)
    #     tmp = df.groupby(['x_at_mod', 'y_at_mod']).apply(group_min).reset_index(drop=True)
    # else:
    #     if sum(np.abs(np.array(angles_that_wraps_into_itself) - abs(sinogram_dict['angle_by_std'])) < angle_close_to_wrap):
    #         idx=df.groupby(['x_at_mod', 'y_at_mod']).axis_root_distance.idxmin()  # take the original data
    #     else:
    #         idx = df.groupby(['x_at_mod', 'y_at_mod']).major_distance.idxmin()
    #     tmp = df.loc[idx]
    tmp_first_level_column = tmp.columns.to_series().replace(['x_at_mod', 'y_at_mod', 'x_center', 'y_center'], 'remove').replace(list('XY'), 'recovered').replace(['y_per_x_ratio', 'distance', 'axis_root_distance', 'closest_to_slop'], 'stat').values
    tmp.columns = [tmp_first_level_column, tmp.columns.values]
    data = pd.merge(tmp, data, left_on=[('remove', 'x_at_mod'), ('remove', 'y_at_mod')], right_on=[('after', 'X'), ('after', 'Y')], how='right').T.sort_index().T.drop('remove', axis=1)
    del tmp
    mse = (data.recovered - data.before).pow(2).values.mean()
    if debug:
        import pylab as plt

        # checking if rotation worked
        fig = plt.figure()
        fig.suptitle('rotating multi modulo for finding best match to angle %g' % sinogram_dict['angle_by_std'])
        df.set_index('X').Y.plot(style='.', grid=True, label='original')
        df.set_index('x_after_rotation').y_after_rotation.plot(style='.', grid=True, label='after rotate to 0')
        plt.plot([0, 5], [0, 5 * np.tan(np.deg2rad(sinogram_dict['angle_by_std']))])
        plt.plot([-mod_size / 2, -mod_size / 2, mod_size / 2, mod_size / 2, -mod_size / 2], [-mod_size / 2, mod_size / 2, mod_size / 2, -mod_size / 2, -mod_size / 2])  # plotting the modulo frame
        fig.legend()

        # original and recovered data
        if 1:
            fig = plt.figure()
            fig.suptitle('angle %g, MSE %g' % (sinogram_dict['angle_by_std'], mse))
            data.recovered.set_index('X').Y.plot(style='.', alpha=0.1, label='recovered')
            data.before.set_index('X').Y.plot(style='.', alpha=0.1, label='original')
            fig.legend()
            if mse > 1e-7 or 1:
                plt.show()
            else:
                plt.close('all')
        else:
            import plotly as py
            import cufflinks
            fig = data[['before', 'recovered']].stack(0).reset_index(drop=False).figure(kind='scatter', x='X', y='Y', categories='level_1', size=4)
            py.offline.plot(fig)
    res=dict(rmse=mse ** 0.5,
             error_per=0,
             pearson=pearson,
             A=None,
             angle=sinogram_dict['angle_by_std'],
             cov_ev_anlge=cov_angle,
             cov = str(cov.tolist()))
    return res


def calc_sinogram(x, y, hist_bins=300, quant_size=0, plot=False):
    import numpy as np
    import pandas as pd
    import warnings
    from skimage.transform import radon  # , rescale, iradon, iradon_sart, hough_line
    from scipy import signal

    max_num=max(max(x),max(y))
    if quant_size:
        bins=np.arange(-(max_num/quant_size+1),(max_num/quant_size+1))*quant_size+quant_size*0.5
    if not quant_size or len(bins)>hist_bins:
        bins=np.linspace(-max_num, max_num, hist_bins, endpoint=True)
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins]*2)  # in order to estimate the angle, we need the space to be square
    H = H[::-1].T
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

    # this will tell us the angle. we can also use idxmax on the sinogram,
    # but the best is to find the angle that has the highest std,
    # becaues max is by single line and std is by all lines
    # we cannot take the angle sum, because they all summed to big numbers
    radon_estimated_angle = sinogram.std().idxmax()

    # max_peaking = signal.find_peaks(sinogram.std(), distance=sinogram.shape[1]//100)[0]
    max_peaking = signal.find_peaks(sinogram.std(), distance=4)[0].tolist()  # we want to find peaks so all values next to the peak will dropped
    max_peaking+=[0,sinogram.shape[0]-1]  # adding also the edges because they are not count as peaks
    max_peaking=np.array(max_peaking)
    # print(max_peaking)
    max_peaking=sinogram.std().iloc[max_peaking]
    # max_peaking=max_peaking[~max_peaking.index.isin([-90, 0, -45, 90, 45])]  # we multiply the pattern to the left right up and down so obviously we tend to get those angles
    radon_estimated_angle=max_peaking.idxmax()

    slop=np.tan(np.deg2rad(radon_estimated_angle))
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
        plot_sinogram(sinogram_dict['image'], sinogram_dict['sinogram'], sinogram_dict['angle_by_std'], slop=sinogram_dict['slop'])

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


def plot_sinogram(image, sinogram, angle_by_std, slop):
    import pylab as plt
    hist_bins=(image.shape[0]+1)  # image should be square
    fig, ax = plt.subplots(1, 4, figsize=(12 * 2.1, 4.5 * 2.1))  # , subplot_kw ={'aspect': 1.5})#, sharex=False)
    ax[0].set_title("data")
    ax[0].imshow(image, cmap=plt.cm.Greys_r)
    image_middle=hist_bins // 2
    ax[0].plot([image_middle, image_middle+hist_bins/5], [image_middle, image_middle-slop * hist_bins/5])

    ax[1].set_title("sinogram")
    ax[1].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(-90, 90, 0, hist_bins))

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
    samples, number_of_bins, quant_size=1000, 1024, 0.003
    cov=np.mat([[1, 0.9], [0.9, 1]])
    cov=None
    snr=1000001
    results=pd.DataFrame()
    for i in range(100):
        res=sinogram_method(samples, number_of_bins, quant_size, snr, cov=cov, debug=True)
        results=results.append(pd.Series(res), ignore_index=True)
        # print('rmse %g, angle %g' % (res['rmse'], res['angle']))
        print('rmse %g' % (res['rmse']))
    # quant_size/=10 # so we can see some errors
    print(results.describe())
    # rmse=sinogram_method(samples, number_of_bins, quant_size, snr, cov=cov, debug=True)


def compare_sinogram_and_eigen_vector(quant_size=False, hist_bins=300, snr=10000, samples=1000):
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

    sinogram_dict = int_force.methods.sinogram.calc_sinogram(data.X.values, data.Y.values, hist_bins=hist_bins, plot=False, quant_size=quant_size)

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
            int_force.methods.sinogram.calc_sinogram(data.X.values, data.Y.values, hist_bins=hist_bins, plot=True, quant_size=quant_size)
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
    # inx = pd.MultiIndex.from_product(a, names=['samples', 'hist_bins', 'snr', 'quant_size'])
    a = dict(samples=[10, 100, 1000, 10000], hist_bins=[100, 300], snr=[100, 1000, 10000], quant_size=[0])
    a = dict(samples=[100], hist_bins=[100], snr=[10000], quant_size=[0.2, 0.1, 0.01, 0.0001])
    inx = pd.MultiIndex.from_product(a.values(), names=a.keys())

    df = pd.DataFrame(index=inx).reset_index(drop=False).sample(frac=1)
    df = pd.concat([df] * 10000, ignore_index=True)

    df = df.join(df.progress_apply(lambda row: compare_sinogram_and_eigen_vector(**row.to_dict()), axis=1).apply(pd.Series))
    df.to_csv('angle_error_by_sinogram.csv', header=None, mode='a')
    # df.to_csv('angle_error_by_sinogram.csv', mode='w')
    print(df.head())


if __name__ == '__main__':
    __debug_angle_error_per_number_of_bins_snr_and_samples()
