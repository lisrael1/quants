#TODO:
#i have a few open questions here:
#when each row at A is going to zero it mean that A is not inversable, and if it's not zero, the output of data*A can be up to the data range so it will not be near zero
#if we do q_data*A*A.I it's like q_data and we didn nothing here




from numpy import matrix
from numpy.random import normal,uniform,randint,choice
import numpy as np
import pylab


class data:
	original_data=None
	after_a_data=None
	after_modulo_data=None
	after_quantizer_data=None
	after_inverse_data=None
	a=None
	def random_original_data(inputs,samples,input_max_value,conditional_var):
		original_data=matrix([normal(uniform(-input_max_value,input_max_value),conditional_var,inputs) for i in range(samples)])






#modolu around 0. we will right shift (+mod) it so it will be between 0 and 2*modulo, do modolu and left shift it by -mod
def modOp(num,mod):
    return np.sign(num)*(abs(num)%mod)
    return (num+mod)%(2*mod)-mod
#this function just return the quantizer bins borders
#if we have left and right border and we want to put quantizer with #options
def quantizer(left,right,options):
    delta=1.0*(right-left)/options
    return np.r_[left+delta/2:right:delta]
#this function quantize a number by a giver quantizer with bins
def quantizise(mumber,quants):
    return min(quants, key=lambda x:abs(x-mumber))




def random_a(number_of_inputs):
	min_val=-10
	max_val=11
	ind=0
	while (True):
		a=matrix(randint(min_val,max_val,number_of_inputs**2)).reshape(number_of_inputs,number_of_inputs)
		#we want each row to be close to 0 but not 0 so it will be invertable
		a[:,-1]=matrix([-1*sum(i.A1) for i in a[:,range(number_of_inputs-1)]]+choice([-1,1],number_of_inputs)).T
		#this is not a good way because we will get non invertable matrix:
		#a[:,-1]=matrix([-1*sum(i.A1) for i in a[:,range(number_of_inputs-1)]]).T
		ind+=1
		if(ind>1000):
			print "number of trys to find inversable matrix = "+str(ind)
			print "cannot find inversable a matrix. exit"
			exit()
		#we will random until we have invertable matrix. it's better to check det rather than rank because of noise
		if (np.linalg.det(a)>1e-2):
			if (ind>20):
				print "WARNING:"
				print "number of trys to find inversable matrix = "+str(ind)
			if (np.linalg.det(a*a.I - np.eye(inputs))>0.1):
				print "there is some error because a*a.I is not I"
			if debug: print "A:\n",a
			return a.T










def random_data(inputs,samples,input_max_value,conditional_var):
	data=matrix([normal(uniform(-input_max_value,input_max_value),conditional_var,inputs) for i in range(samples)])
	return data








def run_on_all_matrix(mat,func,*additional_args):
	return matrix([func(i,*additional_args) for i in mat.A1]).reshape(mat.shape)
#def print_steps(*args):
#	if (print_all_steps):
#		print("{}".format(*args))


def data_by_channel(original_data):
	if debug: print "original_data:\n",original_data
	data=original_data


	a=0.1*random_a(inputs)
	#a=random_a(inputs)


	data=data*a
	if debug: print "data after a:\n",data
	data=run_on_all_matrix(data,modOp,modulo_size)
	if debug: print "data after modulo:\n",data
	if (False):#quantizer
		#should be about 10, not 1000 or something like this
		quants=quantizer(-modulo_size,modulo_size,100)
		if debug: print "quants:\n",quants
		data=run_on_all_matrix(data,quantizise,quants)
		if debug: print "data after quantizer:\n",data
	data=data*a.I
	if debug: print "normal mse:\n",np.mean((original_data.A1-data.A1)**2/np.var(original_data))
	print "normal mse:\n",np.mean((original_data.A1-data.A1)**2/np.var(original_data))
	if debug: print "mse:\n",np.mean((original_data.A1-data.A1)**2)
	if debug: print "origital var:\n",np.var(original_data)
	return data





debug=0
inputs=3
samples=4
modulo_size=5
print_all_steps=1
original_data=random_data(inputs,samples=samples,input_max_value=50,conditional_var=0.1)

[data_by_channel(original_data) for modulo_size in np.arange(1,10,0.1)]






#print "a.I:\n",a.I
#print "data*a:\n",data*a
#print "original_data*a:\n",original_data*a
#print " data*a-original_data*a:\n",data*a-original_data*a
#print "sum of abs :\n",sum(abs(i) for i in (data*a-original_data*a).A1)
#print "data*a var:\n",np.var((original_data*a).A1)
#print "normal mse with a:\n",np.mean(((original_data*a).A1-(data*a).A1)**2/np.var(original_data*a))
#print "data*a*a.I:\n",data*a*a.I



