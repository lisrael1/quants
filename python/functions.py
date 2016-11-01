#!/usr/bin/python
from time import time
start_time = time()
from numpy import matrix as m
from numpy import *
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brent,minimize,fmin
from pylab import plot,show,grid,xlabel,ylabel,title,legend,close,savefig,ion,draw,figure,annotate
from time import sleep
from multiprocessing import Pool
from operator import methodcaller
set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=100, suppress=1, nanstr=None, infstr=None, formatter=None)



"""
if the modulo_size here for example is 3, we will 
not change values between -1.5 and 1.5, but 1.6 will become -1.4
"""
def mod(num,modulo_size):
	return m((m(num)+modulo_size/2.0)%(modulo_size)-modulo_size/2.0)

"""
each row is at specific time, each column is an input
at var=0 you will get normal dist around 0 at all inputs
"""
def generate_data(covar,var,inputs,samples):
	if (inputs<1 or int(inputs)!=inputs):
		print "inputs has to be positive integer"
		exit()
	if (inputs==1):
		return m(random.normal(0,covar,samples)).T
	#first input will be the uniform dist and the others will be the normal around it:
	data=m([hstack((i,random.normal(i,sqrt(covar),inputs-1))) for i in random.normal(0,sqrt(var),samples)])
	return data

def add_dither(data,dither_size):
	rows,columns=data.shape
	if dither_size:
		dither=m(random.uniform(0,dither_size,rows*columns).reshape((rows,columns)))
	else:
		dither=zeros(rows*columns).reshape((rows,columns))
	return data+dither,dither

'''
for now, the quantizer is around 0
example code:
	q=quantizer(0.1,4)
	q=quantizer(0.1,4,10)
	q=quantizer(all_quants=[-0.15,-0.05,0.05,0.15])
	print q
'''
class quantizer():
	def __init__(self,bin_size=None,quants_number=None,offset=None,max_val=None,min_val=None,all_quants=None):
		if bin_size!=None and quants_number!=None and offset!=None and max_val!=None and min_val!=None and all_quants!=None:
			self.bin_size=bin_size
			self.quants_number=quants_number
			self.offset=offset
			self.max_val=max_val
			self.min_val=min_val
			self.all_quants=all_quants
		elif bin_size!=None and quants_number!=None and max_val==None and min_val==None:
			self.offset=0
			if offset!=None:
				self.offset=offset
			self.bin_size=bin_size
			self.quants_number=quants_number
			self.min_val=-self.bin_size*(self.quants_number-1)/2.0+self.offset
			self.all_quants=[i*self.bin_size+self.min_val for i in range(quants_number)]
			self.max_val=self.all_quants[-1]
		elif all_quants!=None:
			self.all_quants=all_quants
			self.bin_size=all_quants[1]-all_quants[0]
			self.quants_number=len(all_quants)
			self.max_val=self.all_quants[-1]
			self.min_val=self.all_quants[1]
			self.offset=(self.max_val+self.min_val)/2.0
	def __str__(self):
		print "bin_size:",self.bin_size
		print "quants_number:",self.quants_number
		print "offset:",self.offset
		print "max_val:",self.max_val
		print "min_val:",self.min_val
		print "all_quants:",self.all_quants
		return ""
		
def plot_pdf_quants(quantizer,mu,sigma):
	x=arange(quantizer.min_val-2*quantizer.bin_size,quantizer.max_val+2*quantizer.bin_size,sigma/1000.0)
	plot(x,norm.pdf(x,mu,sigma),label="pdf")
	plot(quantizer.all_quants,zeros(quantizer.quants_number),"D",label="quantizer")
	legend(loc="best")
	error=analytical_error(quantizer.bin_size,quantizer.quants_number,mu,sigma)
	title("#bins="+str(quantizer.quants_number)+", bin size="+str(quantizer.bin_size)+"\nmu="+str(mu)+", sigma="+str(sigma)+", error="+str(error))
	grid()
	show()

"""
this function quantize a number by a given quantizer with bins
"""
def quantizise(numbers,quant_size,number_of_quants):
	if (number_of_quants>1000):
		return numbers
	#taking max and min values to be at the last bins
	max_val=quant_size*number_of_quants/2.0
	#rounding to the quantizer
	q=rint((numbers+max_val)/(1.0*quant_size))*quant_size-max_val
	q=mat([max_val if i>max_val else -max_val if i<-max_val else i for i in q.A1]).reshape(q.shape)
	return q

#def analytical_error(quant_size,number_of_quants,mu,sigma):
#	#def normal_pdf(x,mu,sigma):
#	#	return exp(-(x-mu)**2/(2*(sigma**2)))/sqrt(2*pi*(sigma**2))
#	def quantizise_single(x,quant_size,number_of_quants):
#		#taking max and min values to be at the last bins
#		max_val=quant_size*number_of_quants/2.0
#		#rounding to the quantizer
#		q=rint((x+max_val)/(1.0*quant_size))*quant_size-max_val
#		if q>max_val:
#			q=max_val 
#		if q<-max_val:
#			q=-max_val
#		return q
#	def mse_pdf(x,mu,sigma,quant_size,number_of_quants):
#		#return normal_pdf(x,mu,sigma)*(mod(x,quant_size*number_of_quants/2)-quantizise_single(x,quant_size,number_of_quants))**2
#		return norm(mu,sigma).pdf(x)*(mod(x,quant_size*number_of_quants/2.0)-quantizise_single(x,quant_size,number_of_quants))**2
#	return quad(mse_pdf,-7*sigma,7*sigma,args=(mu,sigma,quant_size,number_of_quants))[0]
	
'''
get quants, mu and sigma and return the analytic error for those values
note that mu should be around 0 because of the mod in the mse_pdf
print analytical_error(2.5,5,0,1)
'''
def analytical_error(quant_size,number_of_quants,mu,sigma):
	q=quantizer(quant_size,number_of_quants)
	print q.bin_size
	#def normal_pdf(x,mu,sigma):
	#	return exp(-(x-mu)**2/(2*(sigma**2)))/sqrt(2*pi*(sigma**2))
	def quantizise_single(x,quantizer):
		#taking max and min values to be at the last bins
		#rounding to the quantizer
		q=rint((x+quantizer.max_val)/(1.0*quantizer.bin_size))*quantizer.bin_size-quantizer.max_val
		if q>quantizer.max_val:
			q=quantizer.max_val 
		if q<quantizer.min_val:
			q=quantizer.min_val
		return q
	def mse_pdf(x,mu,sigma,quantizer):
		mod_on=1
		if mod_on:
			return norm(mu,sigma).pdf(x)*(mod(x,quantizer.max_val)-quantizise_single(x,quantizer))**2
		else:
			return norm(mu,sigma).pdf(x)*((x)-quantizise_single(x,quantizer))**2
	return quad(mse_pdf,-4*sigma,4*sigma,args=(mu,sigma,q))[0]

'''
look for best quantizer by number of quants and normal dist args
example:
	s=100
	q=find_best_quantizer(10,0,s)
	print q
	plot_pdf_quants(q,0,s)
'''
def find_best_quantizer(number_of_quants,mu,sigma):
	bin_size=fmin(analytical_error,sigma,xtol=sigma/(10*float(number_of_quants)),ftol=sigma*100,args=(number_of_quants,mu,sigma)).tolist()[0]
	#bin_size=fmin(analytical_error,sigma,args=(number_of_quants,mu,sigma)).tolist()[0]
	return quantizer(bin_size=bin_size,quants_number=number_of_quants)
	#print brent(analytical_error,args=(1000,0,1))
	#print minimize(analytical_error,(1,0.1),method='TNC',args=(1000,0,1))
	

#data_matrix should be 2D list, not np matrix...
def lowest_y_per_x(data_matrix,x_column,y_column):
	data_matrix=sorted(data_matrix,key=lambda e:e[y_column], reverse=True)
	data_matrix={i[x_column]:i for i in data_matrix}.values()
	return m(data_matrix)


