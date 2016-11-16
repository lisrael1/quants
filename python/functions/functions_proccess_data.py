"""
if the modulo_size here for example is 3, we will
not change values between -1.5 and 1.5, but 1.6 will become -1.4
"""

def matching(*args):
	if 1:
		return Pool().imap_unordered(*args)
	else:
		return map(*args)
def mod_op(num,modulo_size):
	return m((m(num)+modulo_size/2.0)%(modulo_size)-modulo_size/2.0)

"""
each row is at specific time, each column is an input
at var=0 you will get normal dist around 0 at all inputs
"""
def generate_data(covar,var,inputs,samples):
	rand_int = unpack('I', open("/dev/urandom","rb").read(4))[0]
	random.seed(rand_int)
	if (inputs<1 or type(inputs)!=int):
		print "inputs has to be positive integer"
		exit()
	if (inputs==1):
		return m(random.normal(0,covar,samples)).T
	#first input will be the normal dist by covar and the others will be the normal around it by var:
	#(random.normal gets mu and sigma, not the var...
	data=m([hstack([i,random.normal(i,sqrt(var),inputs-1)]) for i in random.normal(0,sqrt(covar),samples)])
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
	def __init__(self,number_of_samples,covar,x_quantizer,y_quantizer=None,number_of_inputs=2,dither_on=1,modulo_on=1):
		self.x_quantizer=x_quantizer
		self.y_quantizer=y_quantizer
		self.number_of_inputs=number_of_inputs
		self.number_of_samples=number_of_samples
		self.covar=covar
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
		print "modulo size\n",self.x_quantizer.modulo_edge_to_edge
		print "number of quants\n",self.x_quantizer.number_of_quants
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
		self.dither_size=0
		if (self.dither_on):
			self.dither_size=self.x_quantizer.bin_size
	def finish_calculations(self):
		self.error=self.original_data-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.mse_per_input_sample=1.0*sum(self.error.A1.T*self.error.A1)/(self.number_of_inputs*self.number_of_samples)
		#what should impact on the mse is the number of inputs, samples modulo size and covar but not on the single data variance
		self.snr=1.0*self.x_quantizer.var/self.mse_per_input_sample
		self.capacity=log2(self.snr+1)
	#this function is for only 2 inputs
	def run_sim(self):
		self.original_data=generate_data(self.covar,self.x_quantizer.var,self.number_of_inputs,self.number_of_samples)
		#TODO here we do x-y, where y is the first column but in fact we need matrix with permutation on all matrix, not just the first column and not just x-y, it can also be 2x+1y etc.
		self.original_y=self.original_data[:,0]
		if (self.number_of_inputs==1):
			self.original_y=m(zeros(self.original_data.shape[0])).T
		self.after_dither,self.dither=add_dither(self.original_data,self.dither_size)
		self.x_after_modulo=mod_op(self.after_dither,self.x_quantizer.modulo_edge_to_edge)
		if (self.number_of_inputs==1 and self.modulo_on==False):
			self.x_after_modulo=self.after_dither
		self.x_after_quantizer=self.x_quantizer.quantizise(self.x_after_modulo)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod_op(self.x_after_quantizer-self.original_y,self.x_quantizer.modulo_edge_to_edge)
		if (self.number_of_inputs==1 and self.modulo_on==False):
			self.x_y_delta=self.x_after_quantizer
		self.recovered_x=self.x_y_delta+self.original_y
		self.finish_calculations()
	def __iter__(self):
		return self.__dict__.iteritems()
	def dict(self):#best way, but you need __iter__ and you done need d()
		"""
			o=pd.DataFrame([i.dict() for i in d])
			o.to_csv("a.csv")
		"""
		#return pd.DataFrame({k:[v] for k,v in OrderedDict(self).iteritems()})
		x_quant={"x_quantizer_"+k:v for k,v in OrderedDict(self.x_quantizer).iteritems() if type(v)==int or type(v)==float or type(v)==float64 or type(v)==bool}
		y_quant={"y_quantizer_"+k:v for k,v in OrderedDict(self.y_quantizer).iteritems() if type(v)==int or type(v)==float or type(v)==float64 or type(v)==bool}
		dt1    ={k:v for k,v in OrderedDict(self).iteritems() if type(v)==int or type(v)==float or type(v)==float64 or (type(v)==matrix and len(v)==1)}
		dt2    ={k:v for k,v in OrderedDict(self).iteritems() if type(v)==matrix and v.shape[0]==1 and v.shape[1]==1}#adding also case when we only run 1 sample
		#return  dict(x_quant.items()+y_quant.items()+dt.items())
		return  OrderedDict(x_quant.items()+y_quant.items()+dt1.items()+dt2.items())

#class data_1_input(data_multi_inputs):

class data_2_inputs(data_multi_inputs):
	def print_inputs(self):
		print "===variables==="
		print "number of inputs\n",self.number_of_inputs
		print "number of samples\n",self.number_of_samples
		print "x modulo size\n",self.x_quantizer.modulo_edge_to_edge
		print "y modulo size\n",self.y_quantizer.modulo_edge_to_edge
		print "number of x quants\n",self.x_quantizer.number_of_quants
		print "number of y quants\n",self.y_quantizer.number_of_quants
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
		self.snr=(4.0*self.x_quantizer.var*self.x_quantizer.var)/self.mse_per_input_sample
		self.capacity=log2(self.snr+1)
	#this function is for only 2 inputs
	def run_sim(self):
		self.original_data=generate_data(self.covar,self.x_quantizer.var,self.number_of_inputs,self.number_of_samples)
		self.original_x=self.original_data[:,1] #[:,1:]
		self.original_y=self.original_data[:,0]
		self.x_after_dither,self.dither=add_dither(self.original_x,self.dither_size)
		self.x_after_modulo=mod_op(self.x_after_dither,self.x_quantizer.modulo_edge_to_edge)
		self.x_after_quantizer=self.x_quantizer.quantizise(self.x_after_modulo)-self.dither
		self.y_after_dither=self.original_y+self.dither
		self.y_after_modulo=mod_op(self.y_after_dither,self.y_quantizer.modulo_edge_to_edge)
		self.y_after_quantizer=self.y_quantizer.quantizise(self.y_after_modulo)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod_op(self.x_after_quantizer-self.y_after_quantizer,self.x_quantizer.modulo_edge_to_edge)
		self.recovered_x=self.x_y_delta+self.y_after_quantizer
		self.finish_calculations()

#a function for running parallel:
def n(a):
	a.run_sim()
	return a


