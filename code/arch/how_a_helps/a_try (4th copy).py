#TODO:
#i have a few open questions here:
#when each row at A is going to zero it mean that A is not inversable, and if it's not zero, the output of data*A can be up to the data range so it will not be near zero
#if we do q_data*A*A.I it's like q_data and we didn nothing here




from numpy import matrix
from numpy.random import normal,uniform,randint,choice
import numpy as np
import pylab
from pylab import show,grid,plot


class data:
	original_data=None
	after_a_data=None
	after_modulo_data=None
	after_quantizer_data=None
	after_inverse_data=None
	a=None
	
	modulo_size=None
	number_of_inputs=None
	mse=None
	var=None
	normalized_mse=None
	def __str__(self):
		st=""
		st+="\n\t number_of_inputs:\n"+str(self.number_of_inputs)
		st+="\n\t modulo_size:\n"+str(self.modulo_size)
		st+="\n\t a:\n"+str(self.a)
		st+="\n\t original data:\n"+str(self.original_data)
		st+="\n\t after_a_data:\n"+str(self.after_a_data)
		st+="\n\t after_modulo_data:\n"+str(self.after_modulo_data)
		st+="\n\t after_quantizer_data:\n"+str(self.after_quantizer_data)
		st+="\n\t after_inverse_data:\n"+str(self.after_inverse_data)
		st+="\n\t normalized_mse:\n"+str(self.normalized_mse)
		return st
	def random_original_data(self,samples,input_max_value,conditional_var):
		#the inputs are uniformly spread on the range
		mean=uniform(-input_max_value,input_max_value)
		#when we have a number, the inputs are normally spead around it
		self.original_data=matrix([normal(mean,conditional_var,self.number_of_inputs) for i in range(samples)])
	def random_a(self):
		min_val=-10
		max_val=11
		ind=0
		while (True):
			self.a=matrix(randint(min_val,max_val,self.number_of_inputs**2)).reshape(self.number_of_inputs,self.number_of_inputs)
			#we want each row to be close to 0 but not 0 so it will be invertable
			self.a[:,-1]=matrix([-1*sum(i.A1) for i in self.a[:,range(self.number_of_inputs-1)]]+choice([-1,1],self.number_of_inputs)).T
			#TODO we want the sums to be not equall! here we have only -1 and 1...
			#self.a[:,-1]=matrix([-1*sum(i.A1) for i in self.a[:,range(self.number_of_inputs-1)]]+matrix(range(1,self.number_of_inputs).T).T
			#this is not a good way because we will get non invertable matrix:
			#self.a[:,-1]=matrix([-1*sum(i.A1) for i in self.a[:,range(self.number_of_inputs-1)]]).T
			ind+=1
			if(ind>1000):
				print "number of trys to find inversable matrix = "+str(ind)
				print "cannot find inversable a matrix. exit"
				exit()
			#we will random until we have invertable matrix. it's better to check det rather than rank because of noise
			if (np.linalg.det(self.a)>1e-2):
				if (ind>20):
					print "WARNING:"
					print "number of trys to find inversable matrix = "+str(ind)
				if (np.linalg.det(self.a*self.a.I - np.eye(self.number_of_inputs))>0.1):
					print "there is some error because a*a^-1 is not I"
				if debug: print "A:\n",self.a
				self.a=0.1*self.a.T
				return
	def data_by_channel(self):
		self.after_a_data=self.original_data*self.a
		self.after_modulo_data=run_on_all_matrix(self.after_a_data,modOp,self.modulo_size)
		if (False):#quantizer
			#should be about 10, not 1000 or something like this
			quants=quantizer(-self.modulo_size,self.modulo_size,100)
			self.after_quantizer_data=run_on_all_matrix(self.after_modulo_data,quantizise,quants)
		#TODO change after modulo to after quantizer
		self.after_inverse_data=self.after_modulo_data*self.a.I
		#self.normalized_mse=np.mean((self.original_data.A1-self.after_inverse_data.A1)**2/np.var(self.original_data))
		self.mse=np.mean((self.original_data.A1-self.after_inverse_data.A1)**2)
		self.var=np.var(self.original_data)
		self.normalized_mse=self.mse/self.var

		if debug: print "normalized mse:\n",self.normalized_mse
		if debug: print "mse:\n",self.mse
		if debug: print "origital var:\n",self.var
		





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


















def run_on_all_matrix(mat,func,*additional_args):
	return matrix([func(i,*additional_args) for i in mat.A1]).reshape(mat.shape)
#def print_steps(*args):
#	if (print_all_steps):
#		print("{}".format(*args))







debug=0
inputs=3
samples=1
print_all_steps=1
d1=data()
d1.modulo_size=5
d1.number_of_inputs=3
d1.random_original_data(samples=samples,input_max_value=50,conditional_var=0.1)
d1.random_a()
d1.data_by_channel()
o=[]
modulo= [5.78]#np.arange(0.5,6,0.001)
for i in modulo:
	d1.modulo_size=i
	d1.data_by_channel()
	o+=[d1.normalized_mse]
print d1.var
print d1.mse
print d1.normalized_mse
print d1
exit()
plot(modulo,o,'--.')
show(o)








#print "a.I:\n",a.I
#print "data*a:\n",data*a
#print "original_data*a:\n",original_data*a
#print " data*a-original_data*a:\n",data*a-original_data*a
#print "sum of abs :\n",sum(abs(i) for i in (data*a-original_data*a).A1)
#print "data*a var:\n",np.var((original_data*a).A1)
#print "normal mse with a:\n",np.mean(((original_data*a).A1-(data*a).A1)**2/np.var(original_data*a))
#print "data*a*a.I:\n",data*a*a.I



