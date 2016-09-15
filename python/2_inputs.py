#!/usr/bin/python

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
from time import time
start_time = time()
from numpy import matrix as m
from numpy import *
from pylab import plot,show,grid,xlabel,ylabel,title,legend,close,savefig,ion,draw,figure
from time import sleep
from multiprocessing import Pool
from operator import methodcaller
set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=100, suppress=1, nanstr=None, infstr=None, formatter=None)



#if the modulo_size here for example is 3, we will 
#not change values between -1.5 and 1.5, but 1.6 will become -1.4
def mod(num,modulo_size):
	return m((m(num)+modulo_size/2.0)%(modulo_size)-modulo_size/2.0)

#each row is at specific time, each column is an input
def generate_data(covar,var,inputs,samples):
	#fist input will be the uniform dist and the others will be the normal around it:
	data=m([hstack((i,random.normal(i,sqrt(covar),inputs-1))) for i in random.normal(0,sqrt(var),samples)])
	return data
def add_dither(data,dither_size):
	rows,columns=data.shape
	if dither_size:
		dither=m(random.uniform(0,dither_size,rows*columns).reshape((rows,columns)))
	else:
		dither=zeros(rows*columns).reshape((rows,columns))
	return data+dither,dither


#this function quantize a number by a given quantizer with bins
def quantizise(mumbers,quant_size,number_of_quants):
	#taking max and min values to be at the last bins
	max_val=quant_size*number_of_quants/2.0
	#rounding to the quantizer
	q=rint((mumbers+max_val)/quant_size)*quant_size-max_val
	q=mat([max_val if i>max_val else -max_val if i<-max_val else i for i in q.A1]).reshape(q.shape)
	return q


class data:
	#when you create data, it will run it and calculate the dither, the data after modulo and after all decoders..
	def __init__(self,number_of_samples,var,covar,x_mod_size,num_quants_for_x,y_mod_size,num_quants_for_y,dither_on):
		self.number_of_inputs=2
		self.number_of_samples=number_of_samples
		self.var=var
		self.covar=covar
		self.x_mod_size=x_mod_size
		self.y_mod_size=y_mod_size
		self.num_quants_for_x=num_quants_for_x
		self.num_quants_for_y=num_quants_for_y
		self.dither_on=dither_on
		self.init_calculations()
	def __str__(self):
		self.print_all()
		return ""
	def print_all(self):
		self.print_inputs()
		self.print_flow()
		return self
	def print_inputs(self):
		print "===variables==="
		print "number of inputs\n",self.number_of_inputs
		print "number of samples\n",self.number_of_samples
		print "x modulo size\n",self.x_mod_size
		print "y modulo size\n",self.y_mod_size
		print "number of x quants\n",self.num_quants_for_x
		print "number of y quants\n",self.num_quants_for_y
		print "covar\n",self.covar
		print "dither size\n",self.dither_size
		return self
	def print_flow(self):
		print "===data==="
		print "original_data\n",self.original_data
		print "y x_original delta (y is the first input):\n",self.original_data-self.original_data[:,0]
		print "x_after_dither\n",self.x_after_dither
		print "x_after_modulo\n",self.x_after_modulo
		print "y_after_modulo\n",self.y_after_modulo
		print "x_after_quantizer\n",self.x_after_quantizer
		print "y_after_quantizer\n",self.y_after_quantizer
		print "modulo on x_y_delta after quantizers\n",self.x_y_delta
		print "recovered_x\n",self.recovered_x
		print "error\n",self.error
		print "mse\n",m(self.mse_per_input_sample)
		#i think i can remove this... 	print "normalize mse:\n",m(self.normal_mse)
		print "-------------------------------------------"
		return self
	def init_calculations(self):
		self.x_quant_size=1.0*self.x_mod_size/self.num_quants_for_x
		self.y_quant_size=1.0*self.y_mod_size/self.num_quants_for_y
		self.dither_size=0
		if (self.dither_on):
			self.dither_size=self.x_quant_size
	def finish_calculations(self):
		#we need to remove the first column because we get its original values
		self.error=self.original_x-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.mse_per_input_sample=sum(self.error.A1.T*self.error.A1)/((self.number_of_inputs-1)*self.number_of_samples)
		#i think i can remove this... self.all_data_var=var(self.original_x) #should be (2*var)^2/12
		#what should impact on the mse is the number of inputs, samples modulo size and covar but not on the single data variance
		#i think i can remove this... 	self.normal_mse=(self.mse_per_input_sample/self.covar)/((self.number_of_inputs-1)*self.number_of_samples)#not working...
		self.snr=(4*self.var*self.var)/self.mse_per_input_sample
		self.capacity=log2(self.snr+1)
	#this function is for only 2 inputs
	def run_sim(self):
		self.original_data=generate_data(self.covar,self.var,self.number_of_inputs,self.number_of_samples)
		self.original_x=self.original_data[:,1] #[:,1:]
		self.original_y=self.original_data[:,0]
		self.x_after_dither,self.dither=add_dither(self.original_x,self.dither_size)
		self.x_after_modulo=mod(self.x_after_dither,self.x_mod_size)
		self.x_after_quantizer=quantizise(self.x_after_modulo,self.x_quant_size,self.num_quants_for_x)-self.dither
		self.y_after_dither=self.original_y+self.dither
		self.y_after_modulo=mod(self.y_after_dither,self.y_mod_size)
		self.y_after_quantizer=quantizise(self.y_after_modulo,self.y_quant_size,self.num_quants_for_y)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod(self.x_after_quantizer-self.y_after_quantizer,self.x_mod_size)
		self.recovered_x=self.x_y_delta+self.y_after_quantizer
		self.finish_calculations()

#a function for running parallel:
def n(a):
	a.run_sim()
	return a














#this one is just a short simulation to check if the flow ok:
if (0):
	d_for_prints=[data(
		#input*samples: 300*40 will take 27 sec, 300*400 4 minutes, and 300*4000 will take 40 minutes, 300*4e4 will get memory error
		number_of_samples=3,
		var=10,
		covar=1,
		x_mod_size=i,
		num_quants_for_x=j,
		y_mod_size=24,
		num_quants_for_y=300,
		dither_on=0
		) for i in [1,4,8] for j in [1,4,300]]
	d=Pool().imap_unordered(n,d_for_prints)
	#d=map(n,d)
	[i.print_all() for i in d]



#preparing the data:
d=[data(
	#input*samples: 300*40 will take 27 sec, 300*400 4 minutes, and 300*4000 will take 40 minutes, 300*4e4 will get memory error
	number_of_samples=400,
	var=10,
	covar=1,
	x_mod_size=i,
	num_quants_for_x=j,
	y_mod_size=24,
	num_quants_for_y=3,
	dither_on=0
	) for i in arange(0.1,16,.05) for j in range(1,40)]

#running on each number of quants:
if (0):
	d=Pool().imap_unordered(n,d)
	o=m([[i.x_mod_size,i.mse_per_input_sample,i.num_quants_for_x] for i in d])
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
	o=[[i.mse_per_input_sample,i.num_quants_for_x] for i in d]
	#o=[[i.capacity,i.num_quants_for_x] for i in d]
	#now we will take only one line per number of bins:
	o=sorted(o,key=lambda e:e[0], reverse=True)
	o={i[1]:[i[0],i[1]] for i in o}.values()
	o=m(o)

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
		d=[data(
			number_of_samples=40,
	 		var=k,
			covar=1,
			x_mod_size=i,
			num_quants_for_x=j,	
			y_mod_size=k*2.5,
			num_quants_for_y=15,
			dither_on=0
			) 
				#for i in arange(0.1,16,.05) for j in range(1,40)
				for i in [8] for j in range(1,40)
			]
		d=Pool().imap_unordered(n,d)
		#d=map(n,d)
		#o=[[i.x_mod_size,i.capacity,i.num_quants_for_x] for i in d]
		#o=[[i.x_mod_size,i.snr,i.num_quants_for_x] for i in d]
		o=[[i.x_mod_size,i.mse_per_input_sample,i.num_quants_for_x] for i in d]
		print m(o)
		#now we will take only one line per number of bins:
		o=sorted(o,key=lambda e:e[1], reverse=True)
		o={i[2]:[i[0],i[1],i[2]] for i in o}.values()
		o=m(o)
		plot(o[:,2],o[:,1],label=k)
	xlabel("bins")
	ylabel("mse")
	title("best mse per bins")
	grid()
	legend(loc="best")
	print "simulation time: ",time() - start_time,"sec"
	show()
print "simulation time: ",time() - start_time,"sec"
