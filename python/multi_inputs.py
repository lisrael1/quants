#!/usr/bin/python

#next TODO:
#quantizer is not quantizing to the right size, meaning from -0.1 to 0.1 it gives -0.5,0.0.5 for only 1 quantizer, 
#mmm, it's because the quantizer is at 0 and it also cut the data to the min/max so it throw it to -+0.5
#but also i dont see any changes at the mse when adding quantizer, need to check also...
#also try to add colors to functions, operands (=+-/*) and brackets, it will be more readable...


"""
system goes like this:
basic system:
	encoder:
		modulo input x
			modulo size by var(x-y)
	decoder 1:
		multiply [x,y] in A 
		modulo
	decoder 2:
		multiply at the inverse of A
full system:
	in equations:
		mod(   q( mod( ax + D ))    -D - ay)
			a is alpha
		q(ax+D)=ax+D+z   
			z=quantization error
		so we will get the delta between x and y is:
			ax+D+z-D-ay
			=a(x-y+z)
		and for recovering x we will just do delta+y
	in flow:
		input generator:
			by cov matrix
		encoder 
			multipy by alpha
			add dither
			modulo
			quantisize
		the decoder has 2 blocks:
		decoder 1:
			subtitude by dither 
			subtitude by alpha y 
				or any other integer combinations - multiply by A (here its [1,-1]) so it will be inside the modulo
				to get linear algebra we probably better use A=[[1,-1],[-1,1]]
			modulo
			multiply by alpha

			here we actually get mod(x-y) and we hope that it's inside the modulo
		decoder 2:
			add y (the inverse of A)


alpha is var(x-y) / [var(x-y)+var(dither)] - wielner coefficient

opens:
	modulo depends on the statistice - on the cov so it might be that we dont have the same statistics but we want a known modulo size ahead so we want same modulo size to all
	so all inputs have the same modulo size, number of bins
	but what about var and alpha?
	
"""
from numpy import matrix as m
from numpy import *
set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=100, suppress=1, nanstr=None, infstr=None, formatter=None)



#if the modulo_size here for example is 1.5 so we will 
#not change values between -1.5 and 1.5, but 1.6 will become -1.4
def mod(num,modulo_size):
    return m((m(num)+modulo_size)%(2*modulo_size)-modulo_size)


#each row is at specific time, each column is an input
def generate_data(covar,max_val,inputs,samples):
	data=m([random.normal(i,covar,inputs) for i in random.uniform(-max_val,max_val,samples)])
	return data
def add_dither(data,dither_size):
	rows,columns=data.shape
	if dither_size:
		dither=m(random.uniform(0,dither_size,rows*columns).reshape((rows,columns)))
	else:
		dither=zeros(rows*columns).reshape((rows,columns))
	data+=dither
	return data,dither


#this function quantize a number by a given quantizer with bins
def quantizise(mumbers,quant_size,number_of_quants):
	#rounding to the quantizer
	q=rint(mumbers/quant_size)*quant_size
	#taking max and min values to be at the last bins
	max_val=quant_size*number_of_quants
	q=mat([max_val if i>max_val else -max_val if i<-max_val else i for i in q.A1]).reshape(q.shape)
	return q


class data:
	#incriptor_output=[]
	#number_of_bins=0
	#alpha=0
	#dither_on=True
	#modulo_on=True

	#number_of_inputs=0
	#number_of_samples=0
	#covar=0
	#single_data_var=0
	#dither_size=0
	#mod_size=0

	#original_data=[]
	#after_dither=[]
	#dither=0
	#x_after_modulo=[]
	#x_y_delta=[]
	#recovered_x=[]
	#error=[]
	#mse_per_input_sample=0
	#normal_mse=0
	#all_data_var=0	
	#when you create data, it will run it and calculate the dither, the data after modulo and after all decoders..
	def __init__(self,number_of_inputs,number_of_samples,single_data_var,covar,dither_size,mod_size,num_quants):
		self.number_of_inputs,self.number_of_samples,self.single_data_var,self.covar,self.dither_size,self.mod_size,self.num_quants=number_of_inputs,number_of_samples,single_data_var,covar,dither_size,mod_size,num_quants
		self.quant_size=self.mod_size/(self.num_quants+1)
		self.run_sim()
	def finish_calculations(self):
		self.error=self.original_data-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.mse_per_input_sample=sum(self.error.A1.T*self.error.A1)/(self.number_of_inputs*self.number_of_samples)
		self.all_data_var=var(self.original_data) #should be (2*single_data_var)^2/12
		#what should impact on the mse is the number of inputs, samples modulo size and covar but not on the single data variance
		self.normal_mse=(self.mse_per_input_sample/self.covar)/(self.number_of_inputs*self.number_of_samples)#not working...
		
	def prnt(self):
		print "original_data\n",self.original_data
		print "x_after_modulo\n",self.x_after_modulo
		print "x_after_quantizer\n",self.x_after_quantizer
		print "recovered_x\n",self.recovered_x
		print "error\n",self.error
		print "y x original delta:\n",self.original_data-self.original_data[:,0]
		print "mse\n",m(self.mse_per_input_sample)
		print "normalize mse:\n",m(self.normal_mse)
		print "-------------------------------------------"
	def __str__(self):
		self.prnt()
		return ""
	def run_sim(self):
		self.original_data=generate_data(self.covar,self.single_data_var,self.number_of_samples,self.number_of_inputs)
		self.after_dither,self.dither=add_dither(self.original_data,self.dither_size)
		self.x_after_modulo=mod(self.after_dither,self.mod_size)
		self.x_after_quantizer=quantizise(self.x_after_modulo,self.quant_size,self.num_quants)
		self.x_y_delta=mod(self.x_after_quantizer-self.original_data[:,0],self.mod_size)-self.dither
		self.recovered_x=self.x_y_delta+self.original_data[:,0]
		self.finish_calculations()
		return self


#d=[data(number_of_inputs=300,
#		number_of_samples=400,
#		single_data_var=10,
#		covar=0.1,
#		dither_size=0.1,
#		mod_size=0.1,
#		num_quants=2) for i in range(20)]

#da=m([[i.mse_per_input_sample] for i in d])
#m([i.prnt() for i in d])
#print mean(da[:,1])
#print da
#print sum(da)

d=[data(number_of_inputs=3,
		number_of_samples=4,
		single_data_var=10,
		covar=0.1,
		dither_size=i,
		mod_size=0.1,
		num_quants=1).prnt() for i in [0.001,0.01,0.1,1,10]]

