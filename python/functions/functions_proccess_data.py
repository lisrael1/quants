"""
if the modulo_size here for example is 3, we will 
not change values between -1.5 and 1.5, but 1.6 will become -1.4
"""
def mod_op(num,modulo_size):
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

	

#data_matrix should be 2D list, not np matrix...
def lowest_y_per_x(data_matrix,x_column,y_column):
	data_matrix=sorted(data_matrix,key=lambda e:e[y_column], reverse=True)
	data_matrix={i[x_column]:i for i in data_matrix}.values()
	return m(data_matrix)









class data_multi_inputs():
	#when you create data, it will run it and calculate the dither, the data after modulo and after all decoders..
	def __init__(self,number_of_samples,var,covar,mod_size,num_quants,number_of_inputs=2,y_mod_size=-1,num_quants_for_y=-1,dither_on=1,modulo_on=1):
		self.number_of_inputs=number_of_inputs
		self.number_of_samples=number_of_samples
		self.var=var
		self.covar=covar
		self.mod_size=mod_size
		self.y_mod_size=y_mod_size
		self.num_quants_for_y=num_quants_for_y
		self.num_quants=num_quants
		self.dither_on=dither_on
		self.modulo_on=modulo_on
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
		print "modulo size\n",self.mod_size
		print "number of quants\n",self.num_quants
		print "covar\n",self.covar
		print "dither size\n",self.dither_size
		return self
	def print_flow(self):
		print "===data==="
		print "original_data\n",self.original_data
		print "y x_original delta (y is the first input):\n",self.original_data-self.original_data[:,0]
		print "after_dither\n",self.after_dither
		print "x_after_modulo\n",self.x_after_modulo
		print "x_after_quantizer\n",self.x_after_quantizer
		print "original_y\n",self.original_y
		print "x_y_delta\n",self.x_y_delta
		print "recovered_x\n",self.recovered_x
		print "error\n",self.error
		print "mse\n",m(self.mse_per_input_sample)
		#i think i can remove this... 	print "normalize mse:\n",m(self.normal_mse)
		print "-------------------------------------------"
		return self
	def init_calculations(self):
		self.quant_size=1.0*self.mod_size/(self.num_quants+1)
		self.y_quant_size=1.0*self.y_mod_size/self.num_quants_for_y
		self.dither_size=0
		if (self.dither_on):
			self.dither_size=self.quant_size
	def finish_calculations(self):
		self.error=self.original_data-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.mse_per_input_sample=1.0*sum(self.error.A1.T*self.error.A1)/(self.number_of_inputs*self.number_of_samples)
		#what should impact on the mse is the number of inputs, samples modulo size and covar but not on the single data variance
		self.snr=1.0*self.var/self.mse_per_input_sample
		self.capacity=log2(self.snr+1)
	#this function is for only 2 inputs
	def run_sim(self):
		self.original_data=generate_data(self.covar,self.var,self.number_of_inputs,self.number_of_samples)
		#TODO here we do x-y, where y is the first column but in fact we need matrix with permutation on all matrix, not just the first column and not just x-y, it can also be 2x+1y etc.
		self.original_y=self.original_data[:,0]
		if (self.number_of_inputs==1):
			self.original_y=m(zeros(self.original_data.shape[0])).T
		self.after_dither,self.dither=add_dither(self.original_data,self.dither_size)
		self.x_after_modulo=mod_op(self.after_dither,self.mod_size)
		if (self.number_of_inputs==1 and self.modulo_on==False):
			self.x_after_modulo=self.after_dither
		self.x_after_quantizer=quantizise(self.x_after_modulo,self.quant_size,self.num_quants)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod_op(self.x_after_quantizer-self.original_y,self.mod_size)
		if (self.number_of_inputs==1 and self.modulo_on==False):
			self.x_y_delta=self.x_after_quantizer
		self.recovered_x=self.x_y_delta+self.original_y
		self.finish_calculations()
#class data_1_input(data_multi_inputs):

class data_2_inputs(data_multi_inputs):
	def print_inputs(self):
		print "===variables==="
		print "number of inputs\n",self.number_of_inputs
		print "number of samples\n",self.number_of_samples
		print "x modulo size\n",self.mod_size
		print "y modulo size\n",self.y_mod_size
		print "number of x quants\n",self.num_quants
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
	def finish_calculations(self):
		#we calcualte the error only on x, not on y...
		self.error=self.original_x-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.mse_per_input_sample=1.0*sum(self.error.A1.T*self.error.A1)/self.number_of_samples
		#i think i can remove this... self.all_data_var=var(self.original_x) #should be (2*var)^2/12
		#what should impact on the mse is the number of inputs, samples modulo size and covar but not on the single data variance
		#i think i can remove this... 	self.normal_mse=(self.mse_per_input_sample/self.covar)/((self.number_of_inputs-1)*self.number_of_samples)#not working...
		self.snr=(4.0*self.var*self.var)/self.mse_per_input_sample
		self.capacity=log2(self.snr+1)
	#this function is for only 2 inputs
	def run_sim(self):
		self.original_data=generate_data(self.covar,self.var,self.number_of_inputs,self.number_of_samples)
		self.original_x=self.original_data[:,1] #[:,1:]
		self.original_y=self.original_data[:,0]
		self.x_after_dither,self.dither=add_dither(self.original_x,self.dither_size)
		self.x_after_modulo=mod_op(self.x_after_dither,self.mod_size)
		self.x_after_quantizer=quantizise(self.x_after_modulo,self.quant_size,self.num_quants)-self.dither
		self.y_after_dither=self.original_y+self.dither
		self.y_after_modulo=mod_op(self.y_after_dither,self.y_mod_size)
		self.y_after_quantizer=quantizise(self.y_after_modulo,self.y_quant_size,self.num_quants_for_y)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod_op(self.x_after_quantizer-self.y_after_quantizer,self.mod_size)
		self.recovered_x=self.x_y_delta+self.y_after_quantizer
		self.finish_calculations()
	
#a function for running parallel:
def n(a):
	a.run_sim()
	return a


