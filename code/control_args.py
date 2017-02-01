#!/usr/bin/python
from functions import *





#preparing the data:
a=sim_inputs()

#you better use 4e4 samples and more. 
#4e6 takes 4 hours! 4e5 takes 10 minutes. 4e4 40 sec
if (0):
	#simple curve:
	a.number_of_samples=[4e4,4e6]
	a.dither_on=[0]
	a.num_quants=range(2,40)
	a.create_data()
	a.run_sim()
	#print a.plot_results('num_quants','normalized_mse','dither_on')
	print a.plot_results('num_quants','normalized_mse','number_of_samples')

if (1):
	#simple curve:
	a.number_of_samples=[4e2,4e3]
	a.number_of_samples=[4e4,4e5]
	a.dither_on=[0]
	a.num_quants=range(2,40)
	a.create_data()
	a.run_sim()
	#print a.plot_results('num_quants','normalized_mse','dither_on')
	#print a.plot_results('num_quants','normalized_mse','number_of_samples')
	print a.plot_results('num_quants','mod_size','number_of_samples','normalized_mse')
if (0):
	#check number of samples:
	a.num_quants=range(2,40)
	a.number_of_samples=[4e3,4e4,4.1e4,4e5,4.1e5,4e6]
	a.dither_on=[0]
	a.mod_size=[4.5]
	a.create_data()
	a.run_sim()
	print a.plot_results('num_quants','normalized_mse','number_of_samples')

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




print "simulation time: ",time() - start_time,"sec"
