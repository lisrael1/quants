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
	
	fig=pivot['mean'].figure(xTitle='quant_size', yTitle='rmse', title='rmse per quant sizee, by snr, number of binds and methods')
	py.offline.plot(fig, filename='rmse per quant size snr number of bins and method.html', auto_open=False)

	fig = pivot['std'].figure(xTitle='quant_size', yTitle='rmse', title='rmse per quant sizee, by snr, number of binds and methods')
	py.offline.plot(fig, filename='std per quant size snr number of bins and method.html', auto_open=False)

	exit()
	
	
	
	
	
	best_rmse=pivot.min().unstack(2).unstack(0)
	
	a=best_rmse
	print(a.modulo_method/a.clipping_method)
	a=best_rmse
	print(a.ml_method/a.clipping_method)
