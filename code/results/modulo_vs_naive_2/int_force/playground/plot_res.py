if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	# import pdb
	# from IPython.core.debugger import set_trace
	# pdb.set_trace()
	# pd.set_option("display.max_rows",1000)
	pd.set_option("display.max_columns",1000)
	pd.set_option('expand_frame_repr',False)
	pd.set_option('display.max_colwidth',1000)
	import plotly as py
	import cufflinks
	# import scipy.signal as signal
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
	#df.loc[df.rmse.isna(),'rmse']=df.loc[df.rmse.isna()].mse**0.5
	#df.head()
	
	print('doing pivot')
	pivot=df.pivot_table(columns=['snr','number_of_bins','method'], index='quant_size', values='rmse', aggfunc=['mean', 'std'])
	pivot.to_csv('pivot.csv')

	print('plot rmse')
	fig=pivot['mean'].figure(xTitle='quant_size', yTitle='rmse', title='rmse per quant sizee, by snr, number of binds and methods')
	py.offline.plot(fig, filename='rmse per quant size snr number of bins and method.html', auto_open=False)

	print('plot std')
	fig = pivot['std'].figure(xTitle='quant_size', yTitle='rmse', title='std per quant sizee, by snr, number of binds and methods')
	py.offline.plot(fig, filename='std per quant size snr number of bins and method.html', auto_open=False)

	print('plot dist')
	df['legend'] = df.method + ":" + df.snr.astype(str) + ':' + df.number_of_bins.astype(str)
	if 1:
		fig = df[df.legend.str.endswith(':10000:101')].sample(10000).figure(kind='scatter', x='quant_size', y='rmse', categories='legend', size=4, opacity=0.2, title='all samples per quant sizee, by snr, number of binds and methods')
	else:
		fig = df.figure(kind='scatter', x='quant_size', y='rmse', categories='legend', size=1, opacity=0.01, title='all samples per quant sizee, by snr, number of binds and methods')
		for i in range(len(fig['data'])): fig['data'][i]['visible'] = 'legendonly'
	py.offline.plot(fig, filename='all samples per quant size snr number of bins and method.html', auto_open=False)

	exit()
	
	
	
	
	
	best_rmse=pivot.min().unstack(2).unstack(0)
	
	a=best_rmse
	print(a.modulo_method/a.clipping_method)
	a=best_rmse
	print(a.ml_method/a.clipping_method)
