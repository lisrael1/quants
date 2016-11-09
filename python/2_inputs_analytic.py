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

number_of_quants=10
sigma=1.0
q=find_best_quantizer(number_of_quants,sigma)
q.expected_mse=analytical_error(quantizer_i=q)
print q
q.plot_pdf_quants()
exit()


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



#preparing the data:
max_x_bin_number=20
if 1:#trying all modulo sizes:
	qx=[quantizer(number_of_quants=j,modulo_edge_to_edge=i) for i in arange(0.1,10,.05) for j in range(1,max_x_bin_number)]
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


qy=quantizer(number_of_quants=3,modulo_edge_to_edge=24)
d=[data_2_inputs(
	number_of_samples=4e2,#dont put above 4e5
	covar=1,
	x_quantizer=i,
	y_quantizer=qy,
	dither_on=0
	) for i in qx]
print "simulation time2: ",time() - start_time,"sec"

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
		


#running on best mse for each:
if (1):
	if 1:
		d=Pool().imap_unordered(n,d)
	else:
		d=map(n,d)
	print "simulation time3: ",time() - start_time,"sec"

	
	#take what you need and plot it:
	if 1:
		o=[[i.mse_per_input_sample,i.x_quantizer.number_of_quants] for i in d]
		print "simulation time4: ",time() - start_time,"sec"
		o=lowest_y_per_x(o,1,0)
		print "simulation time5: ",time() - start_time,"sec"

		print o
		plot(o[:,1],o[:,0])
	if 0:
		o=[[i.mse_per_input_sample,i.x_quantizer.modulo_edge_to_edge,i.x_quantizer.number_of_quants] for i in d]
		print "simulation time4: ",time() - start_time,"sec"
		o=lowest_y_per_x(o,2,0)
		#o=[[i.capacity,i.x_quantizer.number_of_quants] for i in d]
		#now we will take only one line per number of bins:
		print "simulation time5: ",time() - start_time,"sec"

		print o
		plot(o[:,2],o[:,1])
	
	xlabel("bins")
	ylabel("mse")
	title("best mse per bins")
	grid()
	show()

#this will loop on options 
if (0):
	for k in [10,41,50]:
		d=[data_2_inputs(
			number_of_samples=40,
	 		var=k,
			covar=1,
			mod_size=i,
			num_quants=j,	
			y_mod_size=k*2.5,
			num_quants_for_y=15,
			dither_on=0
			) 
				#for i in arange(0.1,16,.05) for j in range(1,40)
				for i in [8] for j in range(1,40)
			]
		if 0:
			d=Pool().imap_unordered(n,d)
		else:
			d=map(n,d)
		#o=[[i.mod_size,i.capacity,i.x_quantizer.number_of_quants] for i in d]
		#o=[[i.mod_size,i.snr,i.x_quantizer.number_of_quants] for i in d]
		o=[[i.mod_size,i.mse_per_input_sample,i.x_quantizer.number_of_quants] for i in d]
		print m(o)
		#now we will take only one line per number of bins:
		o=lowest_y_per_x(o,2,1)
		plot(o[:,2],o[:,1],label=k)
	xlabel("bins")
	ylabel("mse")
	title("best mse per bins")
	grid()
	legend(loc="best")
	print "simulation time: ",time() - start_time,"sec"
	show()
print "simulation time: ",time() - start_time,"sec"
