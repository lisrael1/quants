import sys
sys.path.append('../../')
import int_force


def ml_map_method(samples, quant_size, number_of_bins, A=None, snr=1000, dont_look_for_A=True, plot=False):
    '''
        if number of bins is high, we will just run pdf on center of bin
        if medium, we will add quantization noise to cov and run pdf
        if low, will run cdf on each pixel
    :param samples:
    :param quant_size:
    :param number_of_bins:
    :param A:
    :param snr:
    :param dont_look_for_A:
    :param plot:
    :return:
    '''
    if number_of_bins>200:
        return ml_modulo_method_without_quantization_on_pdf(samples=samples, quant_size=quant_size, number_of_bins=number_of_bins, snr=snr)
    import numpy as np
    import pandas as pd
    number_of_modulos=3

    cov = int_force.rand_data.rand_data.rand_cov(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

    mod_size = quant_size*number_of_bins

    data = int_force.rand_data.rand_data.random_data(cov, samples)
    tmp = int_force.methods.methods.sign_mod(data, mod_size)
    recovered = int_force.methods.methods.to_codebook(tmp, mod_size / number_of_bins)
    recovered = int_force.methods.methods.from_codebook(recovered, mod_size / number_of_bins)
    if number_of_bins<20:
        shifts = int_force.methods.ml_modulo.ml_map_by_cdf(cov, number_of_bins, mod_size, number_of_modulos=number_of_modulos, plots=plot)
    else:
        shifts = ml_map_by_pdf_with_adding_quant_to_cov(cov, number_of_bins, mod_size, number_of_modulos=number_of_modulos, plots=plot)

    shifts.index.names = ['X', 'Y']
    shifts = shifts.reset_index(drop=False)
    recovered = pd.merge(recovered.round(8), shifts.round(8), on=['X', 'Y'], how='left')
    if recovered.isnull().values.sum():
        print(' ERROR - merged of shifts and recovered has nan. probably because you used even number of bins')
    recovered['new_x'] = recovered.X + recovered.x_shift * mod_size
    recovered['new_y'] = recovered.Y + recovered.y_shift * mod_size
    recovered = recovered[['new_x', 'new_y']]

    recovered.columns = [['recovered'] * 2, ['X', 'Y']]
    tmp.columns = [['after'] * 2, ['X', 'Y']]
    data.columns = [['before'] * 2, ['X', 'Y']]

    data = data.join(tmp).join(recovered)

    tmp=data.before-data.recovered
    tmp.columns = [['error'] * 2, ['X', 'Y']]
    data = data.join(tmp)
    rmse=(data.recovered - data.before).pow(2).values.mean() ** 0.5

    if plot:
        print('rmse %g'%rmse)
        import plotly as py
        import cufflinks
        data_plot = data.stack(0).reset_index(drop=False)
        fig = data_plot.figure(kind='scatter', x='X', y='Y', categories='level_1', size=4)
        py.offline.plot(fig, auto_open=True, filename='data.html')

    res=dict(rmse=rmse,
             error_per=((data.recovered - data.before).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=pearson,
             A=A,
             samples=samples,
             cov = str(cov.tolist()))
    return res


def ml_map_by_pdf_with_adding_quant_to_cov(cov, number_of_bins, mod_size, number_of_modulos=7, plots=False, debug=False):
    import numpy as np
    import pandas as pd
    from scipy.stats import multivariate_normal

    pd.set_option("display.max_columns", 1000)  # don’t put … instead of multi columns
    pd.set_option('expand_frame_repr', False)  # for not wrapping columns if you have many
    pd.set_option("display.max_rows", 30)
    pd.set_option('display.max_colwidth', 1000)

    bin_size=mod_size/number_of_bins
    cov[0,0]+=bin_size**2/12
    cov[1,1]+=bin_size**2/12
    df=bins_edges_with_duplication_of_modulo(rounding=10, mod_size=mod_size, number_of_bins=number_of_bins, number_of_modulos=5)
    df['pdf'] = multivariate_normal.pdf(df[['x_center', 'y_center']].values, mean=[0, 0], cov=cov)

    probability_shifts = df.pivot_table(index=['x_modulo_shifts', 'y_modulo_shifts'], columns=['x_mod', 'y_mod'], values='pdf').idxmax().unstack()
    x_shift = probability_shifts.applymap(lambda x: x[0])
    y_shift = probability_shifts.applymap(lambda x: x[1])
    ml=x_shift.stack().to_frame('x_shift').join(y_shift.stack().to_frame('y_shift'))
    return ml


def bins_edges_with_duplication_of_modulo(rounding=10, mod_size=10, number_of_bins=100, number_of_modulos=5):
    import itertools
    import numpy as np
    import pandas as pd

    '''first generating the map with all the pixels, then each pixel will get it's cdf'''
    bin_size=mod_size/number_of_bins
    bin_edges=np.linspace(-(number_of_modulos+0.5)*mod_size, (number_of_modulos+0.5)*mod_size, (2*number_of_modulos+1)*number_of_bins+1, endpoint=True)
    # bin_edges=np.round(bin_edges, rounding)
    bin_centers=(bin_edges[1:]+bin_edges[:-1])/2
    if 0:
        df=pd.DataFrame(list(itertools.product(*[bin_centers] * 2)), columns='x_center,y_center'.split(','))
    else:
        a = pd.DataFrame(columns=bin_centers, index=bin_centers)
        b = a.fillna(0).stack().reset_index(drop=False).drop(0, axis=1)
        b.columns='x_center,y_center'.split(',')
        df=b
    df=df.join(int_force.methods.methods.sign_mod(df, mod_size).rename(columns=dict(x_center='x_mod', y_center='y_mod')))
    df['x_modulo_shifts']=round((df.x_center-df.x_mod)/mod_size)  # x_center is the original data, before modulo, and x_mod is after. we want to see how much modulo shifting we had
    df['y_modulo_shifts']=round((df.y_center-df.y_mod)/mod_size)
    df['x_low']=df.x_center-bin_size/2
    df['y_low']=df.y_center-bin_size/2
    df['x_high']=df.x_center+bin_size/2
    df['y_high']=df.y_center+bin_size/2
    if rounding:
        df=df.round(rounding)
    modulo_group=df.groupby(['x_modulo_shifts', 'y_modulo_shifts']).size().reset_index().reset_index().drop(0, axis=1).rename(columns=dict(index='modulo_group_number'))
    df=pd.merge(df, modulo_group, on=['x_modulo_shifts', 'y_modulo_shifts'], how='left')
    if df.modulo_group_number.value_counts().sort_values().std()!=0:
        print('WARNING - each modulo group should have %d instances, and instead we have:'%(bin_size**2))
        print(df.modulo_group_number.value_counts().sort_values().describe())
    return df


def ml_map_by_cdf(cov, number_of_bins, mod_size, number_of_modulos=7, plots=False, debug=False):
    '''

    :param cov:
    :param number_of_bins:
    :param mod_size:
    :param number_of_modulos: number of multiply for each side. at 2 we will get multiply of 5X5
    :param plots:
    :param debug:
    :return:
    '''
    import numpy as np
    import pandas as pd
    import itertools

    from scipy.stats import multivariate_normal

    pd.set_option("display.max_columns", 1000)  # don’t put … instead of multi columns
    pd.set_option('expand_frame_repr', False)  # for not wrapping columns if you have many
    pd.set_option("display.max_rows", 30)
    pd.set_option('display.max_colwidth', 1000)

    rounding=10
    df=bins_edges_with_duplication_of_modulo(rounding=rounding, mod_size=mod_size, number_of_bins=number_of_bins, number_of_modulos=number_of_modulos)
    '''now df has pixels with center and edges for each'''
    if plots:
        print('doing cdf')
    if 1:
        rv = multivariate_normal([0, 0], cov)
        if 1:
            if plots: print('starting product')
            cdf=pd.DataFrame(list(itertools.product(*[df[['x_high','x_low']].stack().unique().tolist()] * 2)), columns=list('xy')).round(rounding)  # we do all the unique stuff instead of taking bin_edges because we have floating point issue
            if plots: print('done product')
            '''now calcuating cdf per pixel. each pixel has shared edges with it's neighbors so we will do the cdf offline and then merge it back'''
            if 0:  # cannot do this because at the right upper the cdf will be 1 and the pdf 0...
                cdf['cdf']=0  # for saving cdf calculation, that is slower... we will only calculate cdf on ones that their pdf is high
                cdf['pdf']=multivariate_normal.pdf(cdf[['x', 'y']].values, mean=[0, 0], cov=cov)
                cdf['pdf']=1
                cdf.loc[cdf.pdf>1e-10, 'cdf']=rv.cdf(cdf[cdf.pdf>1e-10][['x', 'y']].values)
                if plots:
                    print('doing cdf on {cdf:,} from total of {total:,} rows'.format(cdf=cdf[cdf.pdf>1e-10].shape[0], total=cdf.shape[0]))
                cdf=cdf.drop('pdf', axis=1)
            cdf['cdf']=rv.cdf(cdf[['x', 'y']].values)

            if plots: print('starting merging results')
            if 0:  # merging is slow, we better use join
                df['high_cdf'] = pd.merge(df[['x_high', 'y_high']].round(rounding), cdf.round(rounding), left_on=['x_high', 'y_high'], right_on=list('xy'), how='left').cdf.values
                df['low_cdf'] = pd.merge(df[['x_low', 'y_low']].round(rounding), cdf.round(rounding), left_on=['x_low', 'y_low'], right_on=list('xy'), how='left').cdf.values
                df['left_cdf'] = pd.merge(df[['x_low', 'y_high']].round(rounding), cdf.round(rounding), left_on=['x_low', 'y_high'], right_on=list('xy'), how='left').cdf.values
                df['down_cdf'] = pd.merge(df[['x_high', 'y_low']].round(rounding), cdf.round(rounding), left_on=['x_high', 'y_low'], right_on=list('xy'), how='left').cdf.values
            else:
                def merging(left, right):
                    '''
                        pd.merge is slow, you better use join
                        and dont do sort_index. it takes time more than it helps join
                        right should be bigger than left, becaues left is only upper lower etc.
                    :param left:
                    :param right:
                    :return:
                    '''
                    col = list('xy')
                    left.columns = col
                    left.set_index(col, inplace=True)
                    if 0:
                        m=left.index.to_frame().reset_index(drop=True).merge(right.index.to_frame().reset_index(drop=True), on=list('xy'), indicator=True, how='outer', suffixes=['','_'])
                        if not m[m._merge=='left'].sort_values(by=['x','y']).empty:
                            print(m[m._merge=='left'].sort_values(by=['x','y']))
                            print(m._merge.value_counts())
                            # print(left.join(right, how='outer').loc[left.join(right, how='outer').isna().cdf.values])
                            # print(pd.concat([left.index.to_frame(),right.index.to_frame()]).drop_duplicates(keep=False).reset_index(drop=True).sort_values(by=['x','y']))
                    merged=left.join(right, how='left').cdf
                    if merged.isna().sum():
                        print('we have nan at the merged step')
                    return merged.values

                cdf=cdf.set_index(list('xy'))
                df['high_cdf'] = merging(df[['x_high', 'y_high']], cdf)
                df['low_cdf'] =  merging(df[['x_low',  'y_low' ]], cdf)
                df['left_cdf'] = merging(df[['x_low',  'y_high']], cdf)
                df['down_cdf'] = merging(df[['x_high', 'y_low' ]], cdf)
            if df.applymap(np.isnan).sum().sum():
                print('WARNING - we have nan values after merging cdf values, and it should be!')
                print('found {nans:,} nans from total of {total:,} cells'.format(nans=df.applymap(np.isnan).sum().sum(), total=df.size))
            if plots: print('done merging results')
        else:  # more time, because we calculate the same dot 4 times
            df['high_cdf']=rv.cdf(df[['x_high', 'y_high']].values)
            df['low_cdf']=rv.cdf(df[['x_low', 'y_low']].values)
            df['left_cdf']=rv.cdf(df[['x_low', 'y_high']].values)
            df['down_cdf']=rv.cdf(df[['x_high', 'y_low']].values)
        df['bin_cdf']=df.high_cdf-df.left_cdf-df.down_cdf+df.low_cdf
    else:  # at bins >10 it's slower because it's not vector operation
        from scipy.stats import mvn
        df['bin_cdf']=df.apply(lambda row:mvn.mvnun(row[['x_low','y_low']].values,row[['x_high','y_high']].values,[0,0],cov.tolist())[0], axis=1)
    if plots: print('done cdf')

    if plots: print('''finding best group at the main modulo''')
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
    pvt=df.pivot_table(index='modulo_group_number', columns=['x_mod', 'y_mod'], values='bin_cdf')
    probability_map_max = pvt.max().unstack()
    if pvt.count().unstack().std().std():
        print('WARNING - probably map modulo didnt worked correctly')
        # df.modulo_group_number.value_counts().sort_values()
    if pvt.applymap(np.isnan).sum().sum():
        print('WARNING - you have nan cells after the pivot, it means that each group has its own x_mod value, which cannot be, unless you have float precision issue')
    if debug:
        print('group_occurrence')
        group_occurrence = 100 * probability_map.stack().value_counts() / probability_map.size
        group_occurrence = group_occurrence.to_frame('percentages').reset_index().rename(columns=dict(index='modulo_group_number'))
        group_occurrence = pd.merge(modulo_group, group_occurrence, on='modulo_group_number', how='right').sort_values('percentages', ascending=False)
        print(group_occurrence)

    if plots:
        print('starting plotting')
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


def ml_modulo_method_by_pdf_on_all_quant_options(samples, number_of_bins, quant_size, snr, A_rows=None, A=None, cov=None, debug=False):
    '''
        the problem is that for each sample, we calculate the pdf for all this sample options,
        like shifting it left 2 modulo size, and up 1 modulo size, and we have a lot of options
        for each sample, we take the pdf of the whole pixel, not just the point itself
        so if you have a lot of samples, this method is very slow,
        but i need to find a faster method for cdf. the one here is slow
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


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    for i in range(10):
        print(ml_map_method(samples=1000, quant_size=0.1, number_of_bins=109, snr=1000, plot=False)['rmse'])

