#!/usr/bin/python
from functions_proccess_data import *
#next TODO:
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
		independed_var=1,
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
d=[data_2_inputs(
	#input*samples: 300*40 will take 27 sec, 300*400 4 minutes, and 300*4000 will take 40 minutes, 300*4e4 will get memory error
	number_of_samples=400,
	var=10,
	independed_var=1,
	mod_size=i,
	num_quants=j,
	y_mod_size=24,
	num_quants_for_y=3,
	dither_on=0
	) for i in arange(0.1,16,.05) for j in range(1,40)]

#running on each number of quants:
if (0):
	d=Pool().imap_unordered(n,d)
	o=m([[i.mod_size,i.normalized_mse,i.num_quants] for i in d])
	for j in set(o[:,2].A1):
		j=int(j)
		print j
		specific=m([i for i in o.tolist() if i[2]==j])
		plot(specific[:,0],specific[:,1])
		xlabel("mod size")
		ylabel("mse")
		title("number of quants ="+str(j))
		grid()
		savefig("mse per modulo/mse vs mod size at "+str(j)+" bins.jpg")
		close()
		


#running on best mse for each:
if (0):
	d=Pool().imap_unordered(n,d)
	#d=map(n,d)
	o=[[i.normalized_mse,i.num_quants] for i in d]
	#o=[[i.capacity,i.num_quants] for i in d]
	#now we will take only one line per number of bins:
	o=lowest_y_per_x(o,0,1)

	print o
	plot(o[:,1],o[:,0])
	xlabel("bins")
	ylabel("mse")
	title("best mse per bins")
	grid()
	print "simulation time: ",time() - start_time,"sec"
	show()

#this will loop on options 
if (1):
	for k in [10,41,50]:
		d=[data_2_inputs(
			number_of_samples=40,
	 		var=k,
			independed_var=1,
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
		#o=[[i.mod_size,i.capacity,i.num_quants] for i in d]
		#o=[[i.mod_size,i.snr,i.num_quants] for i in d]
		o=[[i.mod_size,i.normalized_mse,i.num_quants] for i in d]
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
