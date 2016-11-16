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



#this one is just a short simulation to check if the flow ok:
if (0):
	d_for_prints=[data_2_inputs(
		#input*samples: 300*40 will take 27 sec, 300*400 4 minutes, and 300*4000 will take 40 minutes, 300*4e4 will get memory error
		number_of_samples=3,
		var=10,
		covar=1,
		mod_size=i,
		num_quants=j,
		y_mod_size=24,
		num_quants_for_y=300,
		dither_on=0
		) for i in [1,4,8] for j in [1,4,300]]
	d=Pool().imap_unordered(n,d_for_prints)
	#d=map(n,d)
	[i.print_all() for i in d]




#preparing the data - try to fix this to search for the best one:
if 1:
	print "simulation time start creating quantizers: ",time() - start_time,"sec"
	max_x_bin_number=20
	if 0:#trying all modulo sizes:
		qx=[quantizer(number_of_quants=j,modulo_edge_to_edge=i) for i in arange(0.1,10,.5) for j in range(1,max_x_bin_number)]
	else:#looking for best quantizer:
		#first find the best modulo size - best quantizer:
		sigma=1.0
		def find_best_quantizer_parallel(number_of_quants):
			return find_best_quantizer(number_of_quants,sigma)
		if 1:
			qx=Pool().imap_unordered(find_best_quantizer_parallel,range(1,max_x_bin_number))
		else:
			#qx=[find_best_quantizer(number_of_quants,sigma) for number_of_quants in range(1,max_x_bin_number)]
			qx=map(find_best_quantizer_parallel,range(1,max_x_bin_number))
		#[str(i) for i in qx]
		#[i.plot_pdf_quants() for i in qx]
		print "simulation time finish finding quantizers: ",time() - start_time,"sec"
	print "simulation time1: ",time() - start_time,"sec"
	if 0:
		qy=[quantizer(number_of_quants=200,bin_size=i) for i in [2,4,5,6,7,8]]
	else:
		qy=[quantizer(number_of_quants=200000,bin_size=2)]
	d=[data_2_inputs(
		number_of_samples=4e4,#dont put above 4e5
		covar=10,
		x_quantizer=i,
		y_quantizer=j,
		dither_on=0
		) for i in qx for j in qy]
	print "simulation time2: ",time() - start_time,"sec"
else:#run just a few samples to see if the flow working ok
	d=[data_2_inputs(	
		number_of_samples=1,#dont put above 4e5
		covar=100,
		x_quantizer=quantizer(number_of_quants=5,modulo_edge_to_edge=6.6),
		y_quantizer=quantizer(number_of_quants=200,bin_size=2),
		dither_on=0
		) for i in range(20)]
		


#running on best mse for each number of quants:
if (1):
	if 1:
		d=Pool().imap_unordered(n,d)
	else:
		d=map(n,d)
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
	o=pd.DataFrame([i.dict() for i in d])
	o=o.sort(columns=[x_plot,y_sort])#sorting from A to Z
	print "data ready,",o.index.size,"lines"
	if o.index.size<100:
		o.transpose().to_csv("all_data.csv")
	else:
		o.to_csv("all_data.csv")
	for i in set(o[plot_threads].tolist()):
		o_i=o.loc[o[plot_threads]==i]
		o_i=o_i.sort(columns=[x_plot,y_sort])#sorting from A to Z
		o_i=o_i.drop_duplicates(subset=x_plot,take_last=False)#take the first one, lowest mse
		plot(o_i[x_plot],o_i[y_plot],label=i)
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
	legend(loc="best", shadow=True, title=plot_threads)
	grid()
	print "simulation time before show: ",time() - start_time,"sec"
	show()

#running on each number of quants:
if (0):
	d=Pool().imap_unordered(n,d)
	o=m([[i.mod_size,i.mse_per_input_sample,i.x_quantizer.number_of_quants] for i in d])
	for j in set(o[:,2].A1):
		j=int(j)
		print j," done"
		specific=m([i for i in o.tolist() if i[2]==j])
		plot(specific[:,0],specific[:,1])
		xlabel("mod size")
		ylabel("mse")
		title("number of quants ="+str(j))
		grid()
		savefig("mse per modulo/mse vs mod size at "+str(j)+" bins.jpg")
		close()
print "simulation time: ",time() - start_time,"sec"
