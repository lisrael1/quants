#!/usr/bin/python
execfile("functions/functions.py")

#next TODO:
'''first TODO - update this file to run with analytic finding on the quantizer instead of trying a lot of quantizers'''
#try to do SNR minus quantization SNR
#add different modulo and quantization also on y
#add simple and short flow for code sanity check
#change max val to sigma and change uniform to normal!!!!!!!!!


"""
we have here only 2 inputs, y and x, and each input has its own modulo size and quantization
we want to see how the change of y modulo and quantization replects on x mse

gvim shortcuts:
zf create folding
za toggle folding state
	
"""

max_x_bin_number=40
min_x_bin_number=3
number_of_samples=4e2
modulo_jumps_resolution=0.5
min_modulo_size=5.5
#preparing the data - try to fix this to search for the best one:
if 1:
	print "simulation time start creating quantizers: ",time() - start_time,"sec"
	if 1:#trying all modulo sizes (at relevant range of 4 to 9):
		qx=[quantizer(number_of_quants=j,modulo_edge_to_edge=i) for i in arange(min_modulo_size,9,modulo_jumps_resolution) for j in range(min_x_bin_number,max_x_bin_number)]
	else:#looking for best quantizer:
		qx=matching(find_best_quantizer_parallel_for_1_sigma,range(min_x_bin_number,max_x_bin_number))
		#qx=[find_best_quantizer(number_of_quants,1) for number_of_quants in range(1,max_x_bin_number)]
		#[str(i) for i in qx]
		#[i.plot_pdf_quants() for i in qx]
		#print [[i.number_of_quants,i.all_quants] for i in qx]
		#exit()

		print "simulation time finish finding quantizers: ",time() - start_time,"sec"
	#now lets pick y quantizer:
	if 0:
		qy=[quantizer(number_of_quants=2000,bin_size=i) for i in [0.5,1,1.5,2,2.5,3,4,5]]
	else:#perfect Y
		qy=[quantizer(number_of_quants=200000,bin_size=1)]#if number_of_quants>1e4 it makes it not under quantizer and with bin_size big it's also not under modulo
	d=[data_2_inputs(
		number_of_samples=number_of_samples,#dont put above 4e5
		independed_var=10,
		x_quantizer=i,
		y_quantizer=j,
		dither_on=0
		) for i in qx for j in qy]
	print "simulation time2: ",time() - start_time,"sec"
else:#run just a few samples to see if the flow working ok
	d=[data_2_inputs(	
		number_of_samples=1,#dont put above 4e5
		independed_var=100,
		x_quantizer=quantizer(number_of_quants=5,modulo_edge_to_edge=6.6),
		y_quantizer=quantizer(number_of_quants=200,bin_size=2),
		dither_on=0
		) for i in range(20)]
		


#running on best mse for each number of quants:
if (1):
	d=matching(n,d)
	print "simulation time3: ",time() - start_time,"sec"
	
	if 1:
		plot_threads="y_quantizer_bin_size"
		x_plot='x_quantizer_number_of_quants'
		y_sort="mse_per_input_sample"
		y_plot=y_sort	
		if 0:
			y_plot="x_quantizer_modulo_edge_to_edge"
	if 0:#spliting by modulo size
		plot_threads="x_quantizer_number_of_quants"
		x_plot='x_quantizer_modulo_edge_to_edge'
		y_plot="mse_per_input_sample"
		y_sort=y_plot#doesnt matter because we dont have duplications at x
	#print [[i.x_quantizer.number_of_quants,i.x_quantizer.all_quants,i.mse_per_input_sample] for i in d]
	#exit()
	o=pd.DataFrame([i.dict() for i in d])
	o=o.sort(columns=[x_plot,y_sort])#sorting from A to Z
	print "data ready,",o.index.size,"lines"
	if o.index.size<100:
		o.transpose().to_csv("all_data.csv")
	else:
		o.to_csv("all_data.csv")
	thread_options=set(o[plot_threads].tolist())
	for i in thread_options:
		o_i=o.loc[o[plot_threads]==i]
		o_i=o_i.sort(columns=[x_plot,y_sort])#sorting from A to Z
		o_i=o_i.drop_duplicates(subset=x_plot,take_last=False)#take the first one, lowest mse
		plot(o_i[x_plot],o_i[y_plot],label=i)
		text(10,0.2,o_i[[x_plot,y_plot]].values)
		print o_i[[x_plot,y_plot]]
		if 0:#for ploting the plot threads in different plots
			xlabel(x_plot)
			ylabel(y_plot)
			title("number of quants ="+str(i))
			grid()
			savefig("mse per modulo/mse vs mod size at "+str(i)+" bins.jpg")
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
