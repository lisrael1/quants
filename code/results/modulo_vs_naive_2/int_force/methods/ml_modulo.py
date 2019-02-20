import sys
sys.path.append('../../')
import int_force


def modulo_method(samples, quant_size, number_of_bins, A=None, snr=1000, dont_look_for_A=True):
    '''
    for modulo method
    example:
        quant_size = 0.01
        number_of_bins = 1001
        cov = rand_cov_1()
        data = random_data(cov, 1000)
        mse = modulo_method(data, quant_size, number_of_bins)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12))
    :param data:
    :param quant_size:
    :param number_of_bins:
    :return:
    '''
    import numpy as np
    import pandas as pd

    cov=int_force.rand_data.rand_data.rand_cov(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    input_data = int_force.rand_data.rand_data.random_data(cov,samples) #data.copy()
    res=dict(rmse=(o - input_data).pow(2).values.mean() ** 0.5,
             error_per=((o-input_data).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=pearson,
             A=A,
             cov = str(cov.tolist()))
    return res


def ml_map(cov, number_of_bins, mod_size, number_of_modulos=7, plots=False, debug=False):
    import numpy as np
    import pandas as pd
    import itertools
    from scipy.stats import multivariate_normal

    bin_size=mod_size/number_of_bins
    rv = multivariate_normal([0, 0], cov)
    bin_edges=np.linspace(-number_of_modulos*mod_size/2, number_of_modulos*mod_size/2, number_of_modulos*number_of_bins+1, endpoint=True)
    bin_centers=(bin_edges[1:]+bin_edges[:-1])/2
    df=pd.DataFrame(list(itertools.product(*[bin_centers] * 2)), columns='x_center,y_center'.split(','))
    df=df.join(int_force.methods.methods.sign_mod(df, mod_size).rename(columns=dict(x_center='x_mod', y_center='y_mod'))).round(12)
    df['x_modulo_shifts']=(df.x_center-df.x_mod)/mod_size
    df['y_modulo_shifts']=(df.y_center-df.y_mod)/mod_size
    df['x_low']=df.x_center-bin_size/2
    df['y_low']=df.y_center-bin_size/2
    df['x_high']=df.x_center+bin_size/2
    df['y_high']=df.y_center+bin_size/2
    # print('doing cdf')
    df['high_cdf']=rv.cdf(df[['x_high', 'y_high']].values)
    df['low_cdf']=rv.cdf(df[['x_low', 'y_low']].values)
    df['left_cdf']=rv.cdf(df[['x_low', 'y_high']].values)
    df['down_cdf']=rv.cdf(df[['x_high', 'y_low']].values)
    # print('done cdf')
    df['bin_cdf']=df.high_cdf-df.left_cdf-df.down_cdf+df.low_cdf
    modulo_group=df.groupby(['x_modulo_shifts', 'y_modulo_shifts']).size().reset_index().reset_index().drop(0, axis=1).rename(columns=dict(index='modulo_group_number'))
    df=pd.merge(df, modulo_group, on=['x_modulo_shifts', 'y_modulo_shifts'], how='left')

    probability_shifts = df.pivot_table(index=['x_modulo_shifts', 'y_modulo_shifts', 'modulo_group_number'], columns=['x_mod', 'y_mod'], values='bin_cdf').idxmax().unstack()
    try:
        probability_map=probability_shifts.applymap(lambda x:x[2])
        x_shift=probability_shifts.applymap(lambda x:x[0])
        y_shift=probability_shifts.applymap(lambda x:x[1])
    except:
        print(probability_shifts)
        print(probability_shifts.nunique())
        print('unknown error. for some reason instead of dictionary, we have float at the content. maybe its when we have only 1 sample?. return')
        return dict()
    probability_map_max = df.pivot_table(index='modulo_group_number', columns=['x_mod', 'y_mod'], values='bin_cdf').max().unstack()
    if df.pivot_table(index='modulo_group_number', columns=['x_mod', 'y_mod'], values='bin_cdf').count().unstack().std().std():
        print('WARINING - probably modulo didnt worked correctly')

    if debug:
        print('group_occurrence')
        group_occurrence = 100 * probability_map.stack().value_counts() / probability_map.size
        group_occurrence = group_occurrence.to_frame('percentages').reset_index().rename(columns=dict(index='modulo_group_number'))
        group_occurrence = pd.merge(modulo_group, group_occurrence, on='modulo_group_number', how='right').sort_values('percentages', ascending=False)
        print(group_occurrence)

    if plots:
        import plotly as py
        import cufflinks
        if debug:
            if 0:
                original_heatmap=df[['x_center','y_center','bin_cdf']][(df.x_center==df.x_mod)&(df.y_center==df.y_mod)].set_index(['x_center','y_center']).unstack()
            else:
                original_heatmap=df[['x_center','y_center','bin_cdf']].set_index(['x_center','y_center']).unstack()
            original_heatmap.columns=original_heatmap.columns.get_level_values(1)
            fig = original_heatmap.figure(kind='heatmap', colorscale='Reds')
            # fig = original_heatmap.figure(kind='surface', colorscale='Reds')
            py.offline.plot(fig, filename='original_heatmap.html')
        if 1:
            fig=probability_map.figure(kind='heatmap', colorscale='Reds')
            py.offline.plot(fig, filename='probability_map.html')
            fig = probability_map_max.figure(kind='heatmap', colorscale='Reds')
            py.offline.plot(fig, filename='probability_map_max.html')
        else:
            probability_map=df.pivot_table(index='modulo_group_number', columns=['x_mod', 'y_mod'], values='bin_cdf').idxmax().sort_values().to_frame('modulo_group_number').astype(str).reset_index()
            fig=probability_map.figure(kind='scatter', x='x_mod', y='y_mod', categories='modulo_group_number')
            py.offline.plot(fig)
    ml=x_shift.stack().to_frame('x_shift').join(y_shift.stack().to_frame('y_shift'))
    return ml


def ml_modulo_method(samples, number_of_bins, quant_size, snr, A_rows=None, A=None, cov=None, debug=False):
    import numpy as np
    import pandas as pd

    mod_size = number_of_bins * quant_size
    if type(cov) == type(None):
        cov = int_force.rand_data.rand_data.rand_cov(snr=snr)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    original_data = int_force.rand_data.rand_data.random_data(cov, samples)  # data.copy()
    shifts=int_force.methods.ml_modulo.ml_map(cov, number_of_bins, mod_size, number_of_modulos=9, plots=False)

    data=int_force.rand_data.rand_data.random_data(cov, 1000)
    tmp=int_force.methods.methods.sign_mod(data, mod_size)
    recovered=int_force.methods.methods.to_codebook(tmp, mod_size/number_of_bins)
    recovered=int_force.methods.methods.from_codebook(recovered, mod_size/number_of_bins)
    shifts.index.names=['X','Y']
    shifts=shifts.reset_index(drop=False)
    shifts=shifts.sort_values(['X', 'Y']).round(8)
    recovered=recovered.sort_values(['X', 'Y']).round(8)
    recovered=pd.merge(recovered, shifts, on=['X', 'Y'], how='left')
    recovered['new_x']=recovered.X+recovered.x_shift*mod_size
    recovered['new_y']=recovered.Y+recovered.y_shift*mod_size
    recovered=recovered[['new_x', 'new_y']]

    recovered.columns=[['recovered']*2, ['X', 'Y']]
    tmp.columns=[['after']*2, ['X', 'Y']]
    data.columns=[['before']*2, ['X', 'Y']]

    data=data.join(tmp).join(recovered)

    error = data.before - data.recovered
    mse = error.pow(2).values.mean()
    rmse = mse ** 0.5

    if 0:
        plot_data=data.stack(0).reset_index(drop=False)
        import plotly as py
        import cufflinks
        fig=plot_data.figure(kind='scatter', x='X', y='Y', categories='level_1', size=4)
        py.offline.plot(fig, auto_open=True, filename='data.html')
    res = dict(rmse=rmse,
               error_per=0,
               pearson=pearson,
               A=np.nan,
               cov=str(cov.tolist()))
    return res


def ml_modulo_method_without_quantization_on_pdf(samples, number_of_bins, quant_size, snr, A_rows=None, A=None, cov=None, debug=False):
    import numpy as np
    import pandas as pd
    import plotly as py
    import cufflinks
    import itertools
    from scipy.stats import multivariate_normal

    modulo_size = number_of_bins * quant_size
    if type(cov)==type(None):
        cov = int_force.rand_data.rand_data.rand_cov(snr=snr)

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    original_data = int_force.rand_data.rand_data.random_data(cov, samples)  # data.copy()
    q = int_force.methods.methods.to_codebook(original_data, quant_size, 0)
    quantized_data = int_force.methods.methods.from_codebook(q, quant_size, 0)
    input_data = int_force.methods.methods.sign_mod(quantized_data, modulo_size)

    input_data['sample_index'] = range(input_data.shape[0])

    max_modulo_jump=5
    grid = np.arange(-max_modulo_jump+1, max_modulo_jump)  # * modulo_size# + modulo_size / 2
    grid = pd.DataFrame(list(itertools.product(grid, grid, list(range(samples)))), columns=['x_mod_area', 'y_mod_area', 'sample_index'])
    grid['x_grid'] = grid.x_mod_area * modulo_size
    grid['y_grid'] = grid.y_mod_area * modulo_size
    grid = pd.merge(grid, input_data, on='sample_index', how='left')
    grid['x_output'] = grid.x_grid + grid.X
    grid['y_output'] = grid.y_grid + grid.Y
    grid['pdf'] = multivariate_normal.pdf(grid[['x_output', 'y_output']].values, mean=[0, 0], cov=cov)
    outputs = grid.iloc[grid.groupby('sample_index').pdf.idxmax().values].set_index('sample_index')
    error = original_data - outputs[['x_output', 'y_output']].rename(columns=dict(x_output='X', y_output='Y'))
    mse = error.pow(2).values.mean()
    rmse = mse ** 0.5

    if debug:
        print(rmse)
        if 1:
            error=input_data[['X','Y']]-outputs[['x_output', 'y_output']].rename(columns=dict(x_output='X', y_output='Y'))
        error['max_error'] = error.apply(np.max, axis=1)
        error['big_error'] = error.max_error > error.max_error.quantile(.90)

        output_with_x_y=outputs.copy()[['x_output', 'y_output']].rename(columns=dict(x_output='X', y_output='Y'))
        fig=output_with_x_y.copy()
        fig['legend']='output'
        fig=pd.concat([fig,input_data[['X','Y']]],axis=0, sort=True).fillna('input')
        fig=pd.concat([fig,original_data],axis=0, sort=True).fillna('original')
        fig=pd.concat([fig,original_data.iloc[error[error.big_error].index.values]],axis=0, sort=True).fillna('big_error_input')
        fig=pd.concat([fig,output_with_x_y.iloc[error[error.big_error].index.values]],axis=0, sort=True).fillna('big_error_output')
        max_val=fig[['X', 'Y']].abs().max().max()
        fig=pd.concat([fig,pd.DataFrame(list(itertools.product(*[[-max_val,max_val]]*2)), columns=['X','Y'])],axis=0, sort=True).fillna('plot_edges')
        fig=fig.figure(kind='scatter', x='X', y='Y', categories='legend', size=8, title="RMSE = %f" % rmse, opacity=0.3)
        py.offline.plot(fig)

        if 0:
            outputs.set_index('x_output').sort_index().y_output.plot(style='.', alpha=0.7)
            input_data.set_index('X').sort_index().Y.plot(style='.', alpha=0.7)
            original_data.set_index('X').sort_index().Y.plot(style='.', alpha=0.7)
            original_data.iloc[error[error.big_error].index.values].set_index('X').sort_index().Y.plot(style='.', alpha=0.7)

            plt.axes().set_title("RMSE %f" % rmse)

            plt.show()
    res = dict(rmse=rmse,
               error_per=0,
               pearson=pearson,
               A=np.nan,
               cov=str(cov.tolist()))
    return res


def calc_sinogram(x, y, bins=300):
    import numpy as np
    import pandas as pd
    import warnings
    from skimage.transform import radon  # , rescale, iradon, iradon_sart, hough_line
    from scipy import signal

    max_num=max(max(x),max(y))
    bins=np.linspace(-max_num, max_num, bins, endpoint=True)
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
    # but the best is to find the angle that has the highest std
    # we cannot take the angle sum, because they all summed to big numbers
    radon_estimated_angle = sinogram.std().idxmax()

    # lets remove the peaks at 0 and 90 degrease
    # max_peaking = signal.find_peaks(sinogram.std(), distance=sinogram.shape[1]//100)[0]
    max_peaking = signal.find_peaks(sinogram.std(), distance=4)[0]
    # print(max_peaking)
    max_peaking=sinogram.std().iloc[max_peaking]
    max_peaking=max_peaking[~max_peaking.index.isin([-90, 0, -45, 90, 45])]  # we multiply the pattern to the left right up and down so obviously we tend to get those angles
    radon_estimated_angle=max_peaking.idxmax()

    slop=np.tan(np.deg2rad(radon_estimated_angle))

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
    return dict(image=H, sinogram=sinogram,
                angle_by_std=radon_estimated_angle,
                angle_by_idxmax=sinogram.stack().idxmax()[1],
                overall_sinogram_std=sinogram.std().std(),
                slop=slop,
                y_avg=y.mean(),
                x_avg=x.mean())


if __name__ == '__main__':
    import numpy as np

    samples, number_of_bins, quant_size=300, 19, 1.971141
    cov=np.mat([[1.252697487948626, 1.3951208577696566], [1.3951208577696566, 1.5559781209306283]])

    samples, number_of_bins, quant_size=300, 19, 0.785235
    cov=None
    snr=1000

    for i in range(10):
        rmse=ml_modulo_method(samples, number_of_bins, quant_size, snr, cov=cov, debug=False)['rmse']
        print(rmse)
    # quant_size/=10 # so we can see some errors
    rmse=ml_modulo_method(samples, number_of_bins, quant_size, snr, cov=cov, debug=True)
