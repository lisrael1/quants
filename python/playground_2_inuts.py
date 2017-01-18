#!/usr/bin/python
execfile("functions/functions.py")

#next TODO:
#try to do SNR minus quantization SNR
#probably you better put a quant at the 0, but then you will always will have odd number of quants


"""
we have here only 2 inputs, y and x, and each input has its own modulo size and quantization
we want to see how the change of y modulo and quantization replects on x mse

gvim shortcuts:
zf create folding
za toggle folding state

"""
best_bin_sizes_at_1_var={1: 0, 2: 0.0616666, 3: 1.64, 4: 1.342, 5: 1.186666, 6: 1.0021428571428579, 7: 0.91375, 8: 0.838888, 9: 0.7565, 10: 0.660454, 11: 0.6333333, 12: 0.57346153846153891, 13: 0.535, 14: 0.5106666, 15: 0.4821875, 16: 0.4529411764705884, 17: 0.434166666, 18: 0.415263157894, 19: 0.38925, 20: 0.36142857142857165, 21: 0.368181818181, 22: 0.34456521739130475, 23: 0.32666666, 24: 0.3184, 25: 0.30692307692307708, 26: 0.28555555555, 27: 0.28392857142857153, 28: 0.27741379310344838, 29: 0.265833333, 30: 0.25983870967741973, 31: 0.245625, 32: 0.2360606060606063, 33: 0.23573529411764715, 34: 0.21785714285714297, 35: 0.22208333333333341, 36: 0.21310810810810832, 37: 0.21381578947368451, 38: 0.20243589743589752, 39: 0.188}

max_x_bin_number=20
min_x_bin_number=2
number_of_samples=4e3#put here 1 and look at all_data.csv to see the flow on single sample
modulo_jumps_resolution=0.5
min_modulo_size=5.5
cov=mat([[1,1],[1,10]])

#preparing the data - try to fix this to search for the best one:
if 1:
	print "simulation time start creating quantizers: ",time() - start_time,"sec"
	if 1:#take known best quantizer instead of looking for them
		qx=[simple_quantizer(number_of_quants=i,bin_size=cov[0,0]*best_bin_sizes_at_1_var[i]) for i in range(min_x_bin_number,max_x_bin_number)]
	if 0:#trying all modulo sizes (at relevant range of 4 to 9):
		qx=[simple_quantizer(number_of_quants=j,bin_size=i/j) for i in arange(min_modulo_size,9,modulo_jumps_resolution) for j in range(min_x_bin_number,max_x_bin_number)]
	if 0:#try just a few numbers to see how the flow runs
		qx=[quantizer(number_of_quants=5,bin_size=6.6/5) for i in range(20)] #put number_of_samples=1 and independed_var=100

	#now lets pick y quantizer:
	if 0:
		qy=[simple_quantizer(number_of_quants=2000,bin_size=i) for i in [0.5,1,1.5,2,2.5,3,4,5]]
	else:#perfect Y
		qy=[simple_quantizer(number_of_quants=200000,bin_size=1)]

	d=[sim_2_inputs(
		number_of_samples=number_of_samples,#dont put above 4e5
		cov=cov,
		x_quantizer=i,
		y_quantizer=j,
		dither_on=0
		) for i in qx for j in qy]
	print "simulation time2: ",time() - start_time,"sec"




#running on best mse for each number of quants:
if (1):
	d=matching(n,d)
	print "simulation time3: ",time() - start_time,"sec"

	if 0:#if we dont want parsing text, and we want the class functionality. note that you dont sort the data here but you will have mse
	   [i.x_quantizer.plot_pdf_quants() for i in d]
	   exit()
	if 1:#basic one, exp plot
		plot_threads="y_quantizer_bin_size"
		x_plot='x_quantizer_number_of_quants'
		y_sort="normalized_mse"
		y_plot=y_sort
		if 0:
			y_plot="x_quantizer_modulo_edge_to_edge"
	if 0:#spliting by x_quantizer_number_of_quants
		plot_threads="x_quantizer_number_of_quants"
		x_plot='x_quantizer_modulo_edge_to_edge'
		y_sort="normalized_mse"
		y_plot=y_sort#doesnt matter because we dont have duplications at x

	o=pd.DataFrame([i.dict() for i in d])#note that here we loose the functions of all classes and we just have data!
	o=o.sort(columns=[x_plot,y_sort])#sorting from A to Z
	print "data ready,",o.index.size,"lines"
	if o.index.size<100:
		o.transpose().to_csv("temp/all_data.csv")#we will see each sample at different column
	else:
		o.to_csv("all_data.csv")
	thread_options=set(o[plot_threads].tolist())
	for i in thread_options:
		o_i=o.loc[o[plot_threads]==i]
		o_i=o_i.sort(columns=[x_plot,y_sort])#sorting from A to Z
		o_i=o_i.drop_duplicates(subset=x_plot,take_last=False)#take the first one, lowest mse
		plot(o_i[x_plot],o_i[y_plot],label=i)
		if len(thread_options)==1:
			text(15,0.,o_i[[x_plot,y_plot]].values)
##		print o_i[[x_plot,y_plot]]
		if 0:#for ploting the plot threads in different plots
			xlabel(x_plot)
			ylabel(y_plot)
			title("number of quants ="+str(i))
			grid()
			savefig("temp"+dlmtr+"mse per modulo"+dlmtr+"mse vs mod size at "+str(i)+" bins"+img_type)
			close()
	xlabel(x_plot)
	ylabel(y_plot)
	title("-")
	if len(thread_options)>1:
		legend(loc="best", shadow=True, title=plot_threads)
	grid()
	print "simulation time before show: ",time() - start_time,"sec"
	show()

print "simulation time: ",time() - start_time,"sec"
