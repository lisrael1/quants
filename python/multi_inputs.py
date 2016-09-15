#!/usr/bin/python
from functions import *

#next TODO:
#try to do SNR minus quantization SNR



"""
gvim shortcuts:
zf create folding
za toggle folding state
"""


#preparing the data:
d=[data_multi_inputs(
	number_of_inputs=4,#input*samples: 300*40 will take 27 sec, 300*400 4 minutes, and 300*4000 will take 40 minutes, 300*4e4 will get memory error
	number_of_samples=40,
	var=10,
	covar=1,
	mod_size=i,
	num_quants=j,
	dither_on=0
	) for i in arange(0.1,16,.05) for j in range(1,40)]


#a function for running parallel:
def n(a):
	a.run_sim()
	return a


#parsing output:

#running on each number of quants:
if (0):
	d=Pool().imap_unordered(n,d)
	o=m([[i.mod_size,i.mse_per_input_sample,i.num_quants] for i in d])
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
if (1):
	d=Pool().imap_unordered(n,d)
	#d=map(n,d)
	o=[[i.mse_per_input_sample,i.num_quants] for i in d]
	#o=[[i.capacity,i.num_quants] for i in d]
	#now we will take only one line per number of bins:
	o=lowest_y_per_x(o,1,0)

	print o
	plot(o[:,1],o[:,0])
	xlabel("bins")
	ylabel("mse")
	title("best mse per bins ")
	grid()
	print "simulation time: ",time() - start_time,"sec"
	show()

#this will loop on options 
if (0):
	for k in [10,20,30]:
		d=[data_multi_inputs(
			number_of_inputs=2,#input*samples: 300*40 will take 27 sec, 300*400 4 minutes, and 300*4000 will take 40 minutes, 300*4e4 will get memory error
			number_of_samples=40,
			var=k,
			covar=1,
			mod_size=i,
			num_quants=j,
			dither_on=0
			) for i in arange(0.1,16,.05) for j in range(1,40)]
		d=Pool().imap_unordered(n,d)
		#d=map(n,d)
		#o=[[i.mod_size,i.capacity,i.num_quants] for i in d]
		o=[[i.mod_size,i.snr,i.num_quants] for i in d]
		#o=[[i.mod_size,i.mse_per_input_sample,i.num_quants] for i in d]
		#now we will take only one line per number of bins:
		o=lowest_y_per_x(o,1,2)
		plot(o[:,2],o[:,1])
	grid()
	print "simulation time: ",time() - start_time,"sec"
	show()
print "simulation time: ",time() - start_time,"sec"
