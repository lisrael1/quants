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



max_x_bin_number=20
#preparing the data - try to fix this to search for the best one:
if 0:
	if 1:#trying all modulo sizes:
		qx=[quantizer(number_of_quants=j,modulo_edge_to_edge=i) for i in arange(0.1,10,.5) for j in range(1,max_x_bin_number)]
	else:#looking for best quantizer:
		#first find the best modulo size - best quantizer:
		sigma=1.0
		if 0:#without parallel:
			qx=[find_best_quantizer(number_of_quants,sigma) for number_of_quants in range(1,max_x_bin_number)]
		else:#by parallel:
			def find_best_quantizer_parallel(number_of_quants):
				return find_best_quantizer(number_of_quants,sigma)
			qx=Pool().imap_unordered(find_best_quantizer_parallel,range(1,max_x_bin_number))
			#[str(i) for i in q]
	print "simulation time1: ",time() - start_time,"sec"


if 1:
	qx=[quantizer(number_of_quants=j,modulo_edge_to_edge=i) for i in arange(0.1,10,.05) for j in range(1,max_x_bin_number)]
	qy=[quantizer(number_of_quants=200,bin_size=i,disable_modulo=True) for i in [0.1,1,1.5,2]]
	d=[data_2_inputs(
		number_of_samples=4e2,#dont put above 4e5
		covar=10,
		x_quantizer=i,
		y_quantizer=j,
		dither_on=0
		) for i in qx for j in qy]
	print "simulation time2: ",time() - start_time,"sec"
else:
	d=[data_2_inputs(	
		number_of_samples=1,#dont put above 4e5
		covar=100,
		x_quantizer=quantizer(number_of_quants=5,modulo_edge_to_edge=6.6),
		y_quantizer=quantizer(number_of_quants=20,modulo_edge_to_edge=5,disable_modulo=True),
		dither_on=0
		) for i in range(20)]
		


#running on best mse for each:
if (1):
	if 1:
		d=Pool().imap_unordered(n,d)
	else:
		d=map(n,d)
	print "simulation time3: ",time() - start_time,"sec"
	o=pd.DataFrame([i.dict() for i in d])
	o=o.sort(columns=['x_quantizer_number_of_quants','mse_per_input_sample'])#sorting from A to Z
	print "data ready,",o.index.size,"lines"
	if o.index.size<100:
		o.transpose().to_csv("all_data.csv")
	else:
		o.to_csv("all_data.csv")
	for i in set(o.y_quantizer_bin_size.tolist()):
		o_i=o.loc[o.y_quantizer_bin_size==i]
		o_i=o_i.sort(columns=['x_quantizer_number_of_quants','mse_per_input_sample'])#sorting from A to Z
		o_i=o_i.drop_duplicates(subset="x_quantizer_number_of_quants",take_last=False)#take the first one, lowest mse
		if 1:
			plot(o_i.x_quantizer_number_of_quants,o_i.mse_per_input_sample,label=i)
		else:
			plot(o_i.x_quantizer_number_of_quants,o_i.x_quantizer_modulo_edge_to_edge,label=i)
	
	xlabel("bins")
	ylabel("mse")
	title("best mse per bins")
	legend(loc="best", shadow=True, title="# y quants")
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
