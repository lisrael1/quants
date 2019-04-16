if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	pd.set_option("display.max_columns",1000)
	pd.set_option('expand_frame_repr',False)
	pd.set_option('display.max_colwidth',1000)
	import plotly as py
	import cufflinks
	from glob import glob
	from tqdm import tqdm


	df=pd.DataFrame()
	alls=[]
	#for f in tqdm(glob(r"*gz")[:10]):
	for f in tqdm(glob(r"*gz")):
		tmp=pd.read_csv(f)#, nrows=200000)
		if 0:
			df=pd.concat([df,tmp], axis=0, sort=True)
		if 0:
			df=df.append(tmp,sort=True)
		if 1:
			alls+=[tmp]
	print('now doint concatenation')
	df=pd.concat(alls, ignore_index=True)
	number_of_samples_at_scatter_plot = 50000

	print('doing pivot')
	pivot=df.pivot_table(columns=['snr','number_of_bins','method','samples'], index='quant_size', values='rmse', aggfunc=['mean', 'std'])
	pivot.to_csv('pivot.csv')

	print('plot rmse')

	def remove_nans_from_plot(fig):
		for trace in range(len(fig['data'])):
			inx = np.where(fig['data'][trace]['y'] == '')
			fig['data'][trace]['x'] = np.delete(fig['data'][trace]['x'], inx)
			fig['data'][trace]['y'] = np.delete(fig['data'][trace]['y'], inx)
		return fig
	fig=pivot['mean'].figure(xTitle='quant_size', yTitle='mean', title='rmse per quant size, by snr, number of binds and methods')
	fig=remove_nans_from_plot(fig)
	py.offline.plot(fig, filename='plot rmse per quant size snr number of bins and method.html', auto_open=False)

	print('plot std')
	fig = pivot['std'].figure(xTitle='quant_size', yTitle='std', title='std per quant size, by snr, number of binds and methods')
	py.offline.plot(fig, filename='plot std per quant size snr number of bins and method.html', auto_open=False)

	print('find cov angle')
	if not 'angle' in df.columns.values:
		import sys
		sys.path.append('../../')
		import int_force
		from tqdm import tqdm
		tqdm.pandas()
		df['angle'] = df[df.method=='sinogram_method'].sample(100000).progress_apply(lambda row: int_force.rand_data.find_slop.get_cov_ev(eval(row['cov']))[1], axis=1)

	def redo_hist_on_hist(hist, number_of_bins = 150):
		'''

		:param hist: must by df
		:param number_of_bins:
		:return:
		'''
		new_index_bin_size = (hist.index.max() - hist.index.min()) / number_of_bins
		hist['grouped_code_error'] = ((hist.reset_index().iloc[:,0] / new_index_bin_size).round() * new_index_bin_size).values
		grouped_error_hist = hist.groupby('grouped_code_error').agg(np.sum)
		return grouped_error_hist
	hist = redo_hist_on_hist(df[~df.angle.isna()].set_index('angle').rmse.to_frame('rmse')).rmse
	hist /= hist.sum()

	fig = hist.figure(kind='scatter', title='rmse per angle')
	py.offline.plot(fig, filename='plot rmse per angle hist.html', auto_open=False)

	print('plot angle error')
	if 'angle' in df.columns:
		df['angle_error']=df.angle-df.cov_ev_angle
		for x, y in [['cov_ev_angle', 'angle_error'], ['cov_ev_angle', 'rmse'], ['angle_error', 'rmse']]:
			fig = df.dropna(subset=['angle']).pivot_table(index=x, columns='samples', values=y).sample(frac=1).head(number_of_samples_at_scatter_plot).stack().to_frame(y).reset_index(drop=False).astype(str).figure(
				kind='scatter', x=x, y=y, categories='samples', size=7, opacity=0.9, title='%s per %s'%(y, x))
			py.offline.plot(fig, filename='plot %s per %s.html'%(y, x), auto_open=False)

	print('plot dist')
	df['legend'] = df.method + ":" + df.snr.astype(str) + ':' + df.number_of_bins.astype(str)
	fig = df[df.legend.str.endswith(':10000:101')]
	if not fig.empty:
		fig = fig.sample(10000).figure(kind='scatter', x='quant_size', y='rmse', categories='legend', size=4, opacity=0.2, title='all samples per quant size, by snr, number of binds and methods')
		py.offline.plot(fig, filename='plot all samples per quant size snr number of bins and method.html', auto_open=False)
