def matching(*args):
	if 1 and not "win" in platform:
		return Pool().imap_unordered(*args)
	else:
		return map(*args)

"""
if the modulo_size here for example is 3, we will
not change values between -1.5 and 1.5, but 1.6 will become -1.4
"""
def mod_op(num,modulo_size):
	return m((m(num)+modulo_size/2.0)%(modulo_size)-modulo_size/2.0)

"""
each row is at specific time, each column is an input
at independed_var=0 you will get normal dist around 0 with var of depended_var, at all inputs
"""
def generate_data(independed_var,depended_var,number_of_inputs,samples):
	if not "win" in platform:#at windows we dont run parallel and we dont have /dev/urandom
	     rand_int = unpack('I', open("/dev/urandom","rb").read(4))[0]
	     random.seed(rand_int)
	if (number_of_inputs<1 or type(number_of_inputs)!=int):
		print "number_of_inputs has to be positive integer"
		exit()
	if (number_of_inputs==1):
		return m(random.normal(0,independed_var,samples)).T
	#first input will be the normal dist by independed_var and the others will be the normal around it by var:
	#(random.normal gets mu and sigma, not the var...
	data=m([hstack([i,random.normal(i,sqrt(depended_var),number_of_inputs-1)]) for i in random.normal(0,sqrt(independed_var),samples)])
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

class sim_2_inputs():
	#when you create data, it will run it and calculate the dither, the data after modulo and after all decoders..
##	def __init__(self,number_of_samples,independed_var,x_quantizer,y_quantizer=None,dither_on=1):
	def __init__(self,number_of_samples,cov,x_quantizer,y_quantizer=None,dither_on=0):
		self.x_quantizer=x_quantizer
		self.y_quantizer=y_quantizer
		self.number_of_samples=number_of_samples
		self.depended_var=cov[0,1]
		self.x_var=cov[0,0]
		self.y_var=cov[1,1]
		self.x_mod=self.x_quantizer.modulo_edge_to_edge
		self.y_mod=self.y_quantizer.modulo_edge_to_edge
		self.dither_on=dither_on
		self.init_calculations()
	def init_calculations(self):
		self.dither_size=0
		if (self.dither_on):
			self.dither_size=self.x_quantizer.bin_size
	def __iter__(self):#just for dict function
		return self.__dict__.iteritems()
	def dict(self):
		"""
			o=pd.DataFrame([i.dict() for i in d])
			o.to_csv("a.csv")
		"""
		#return pd.DataFrame({k:[v] for k,v in OrderedDict(self).iteritems()})
		x_quant={"x_quantizer_"+k:v for k,v in OrderedDict(self.x_quantizer).iteritems() if type(v)==int or type(v)==float or type(v)==float64 or type(v)==bool}
		y_quant={"y_quantizer_"+k:v for k,v in OrderedDict(self.y_quantizer).iteritems() if type(v)==int or type(v)==float or type(v)==float64 or type(v)==bool}
		dt1    ={k:v for k,v in OrderedDict(self).iteritems() if type(v)==int or type(v)==float or type(v)==float64 or (type(v)==matrix and len(v)==1)}
		dt2    ={k:v for k,v in OrderedDict(self).iteritems() if type(v)==matrix and v.shape[0]==1 and v.shape[1]==1}#adding also case when we only run 1 sample
		return  OrderedDict(x_quant.items()+y_quant.items()+dt1.items()+dt2.items())
	def finish_calculations(self):
		#we calcualte the error only on x, not on y...
		self.error=self.original_x-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.error_bias=mean(self.error)
		self.mse=sum(self.error.A1.T*self.error.A1)/self.number_of_samples
		self.error_from_big_errors=m([i for i in self.error.A1 if i>2*self.x_quantizer.bin_size])
		self.number_of_big_errors=self.error_from_big_errors.A1.size
		self.mse_from_big_errors=sum(self.error_from_big_errors.A1.T*self.error_from_big_errors.A1)/self.number_of_samples#self.error_from_big_errors.A1.size
		self.normalized_mse=self.mse*self.x_var
		#what should impact on the mse is the number of inputs, samples modulo size and independed_var but not on the single data variance
		self.snr=self.x_var/(self.normalized_mse)
		self.capacity=log2(self.snr+1)
		self.snr_normalized=1.0/self.normalized_mse
		self.capacity_normalized=log2(self.snr_normalized+1)

	def run_sim(self):
		self.original_data=generate_data(self.x_var,self.depended_var,2,self.number_of_samples)
		self.original_x=self.original_data[:,1] #[:,1:]
		self.original_y=self.original_data[:,0]
		self.x_after_dither,self.dither=add_dither(self.original_x,self.dither_size)
		self.x_after_modulo=mod_op(self.x_after_dither,self.x_mod)
		self.x_after_quantizer=self.x_quantizer.quantizise(self.x_after_modulo)-self.dither
		self.y_after_dither=self.original_y+self.dither
		self.y_after_modulo=mod_op(self.y_after_dither,self.y_mod)
		self.y_after_quantizer=self.y_quantizer.quantizise(self.y_after_modulo)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod_op(self.x_after_quantizer-self.y_after_quantizer,self.x_mod)
		self.recovered_x=self.x_y_delta+self.y_after_quantizer
		self.finish_calculations()

#a function for running parallel:
def n(a):
	a.run_sim()
	return a


