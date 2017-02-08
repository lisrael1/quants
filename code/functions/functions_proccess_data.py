sim_number_id=0
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

def sim_log_print(msg):
	msg=str(datetime.now())+" "+str(time() - start_time)+" "+msg
	if 0:
		print msg
	else:
		f=open(output_log_file,"a")
		f.write(msg+"\n")
		f.close()
	
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
	def __init__(self,number_of_samples,cov,x_quantizer,y_quantizer=None,dither_on=0):
		global sim_number_id
		self.sim_id=sim_number_id
		sim_number_id+=1
		self.x_quantizer=x_quantizer
		self.y_quantizer=y_quantizer
		self.number_of_samples=number_of_samples
		self.cov=cov
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
	def sim_log(self,msg):
		log=str(self.sim_id).zfill(8)+" "+str(msg)
		sim_log_print(log)
	#def __del__(self):
	#   self.sim_log("deleting simulation")
	def __iter__(self):#just for dict function
		return self.__dict__.iteritems()
	def dict(self):
            def test(v):
                "to clean some big datas"
                if type(v)==matrix:
                    return v.A1.size<5
                if type(v)==ndarray:
                    return len(v)<5
                if "simple_quantizer instance" in str(v):#it's also taking the main function of the classes so removing those values
                    return False
                return True
            x_quant={"x_quantizer_"+k:v for k,v in OrderedDict(self.x_quantizer).iteritems() if test(v)}
            y_quant={"y_quantizer_"+k:v for k,v in OrderedDict(self.y_quantizer).iteritems() if test(v)}
            all_variables={k:v for k,v in OrderedDict(self).iteritems() if test(v)}
            return OrderedDict(x_quant.items()+y_quant.items()+all_variables.items())
	def table(self):
            data=self.dict()
            df=pd.DataFrame([data])
            return df
	def finish_calculations(self):
		#we calcualte the error only on x, not on y...
		self.error=self.original_x-self.recovered_x
		#for mse we will flaten the error matrix so we can do power 2 just by dot product
		self.error_bias=mean(self.error)
		self.mse=sum(self.error.A1.T*self.error.A1)/self.number_of_samples
		self.error_from_big_errors=m([i for i in self.error.A1 if i>2*self.x_quantizer.bin_size])
		self.number_of_big_errors=self.error_from_big_errors.A1.size
		self.mse_from_big_errors=sum(self.error_from_big_errors.A1.T*self.error_from_big_errors.A1)/self.number_of_samples#self.error_from_big_errors.A1.size
		self.conditional_var=float(self.cov[0,0])+float(self.cov[1,1])-float(2*self.cov[0,1])
		self.normalized_mse=self.mse/self.conditional_var
		#what should impact on the mse is the number of inputs, samples modulo size and independed_var but not on the single data variance
		self.snr=self.x_var/(self.normalized_mse)
		self.capacity=log2(self.snr+1)
		self.snr_normalized=1.0/self.normalized_mse
		self.capacity_normalized=log2(self.snr_normalized+1)
		self.sim_log("finish calculation on single sim ")

	def run_sim(self):
		if not "win" in platform:#at windows we dont run parallel and we dont have /dev/urandom
		     rand_int = unpack('I', open("/dev/urandom","rb").read(4))[0]
		     random.seed(rand_int)
		self.sim_log("starting calculation")
		#self.original_data=generate_data(self.x_var,self.depended_var,2,self.number_of_samples)
		self.original_data=m(random.multivariate_normal([0,0], self.cov, int(self.number_of_samples)))
		self.sim_log("done generating data for sim number with size of "+str(getsizeof(self.original_data.tolist())))
		self.original_x=self.original_data[:,1] #[:,1:]
		self.original_y=self.original_data[:,0]
		self.x_after_dither,self.dither=add_dither(self.original_x,self.dither_size)
		self.x_after_modulo=mod_op(self.x_after_dither,self.x_mod)
		self.sim_log("starting qunatizing")
		self.x_after_quantizer=self.x_quantizer.quantizise(self.x_after_modulo)-self.dither
		self.sim_log("finish quantizing")
		self.y_after_dither=self.original_y+self.dither
		self.y_after_modulo=mod_op(self.y_after_dither,self.y_mod)
		self.y_after_quantizer=self.y_quantizer.quantizise(self.y_after_modulo)-self.dither
		#TODO alpha
		#TODO modulo and quantizer also on y, but not the same modulo and quantizer of x
		#aqctually we pick here x-y but it should be multiply in A
		self.x_y_delta=mod_op(self.x_after_quantizer-self.y_after_quantizer,self.x_mod)
		self.recovered_x_before_alpha=self.x_y_delta+self.y_after_quantizer
		self.x_samples_var=var(self.original_x)
		self.recovered_x_before_alpha_var=var(self.recovered_x_before_alpha)
		self.alpha=self.x_samples_var/self.recovered_x_before_alpha_var
		self.recovered_x=self.recovered_x_before_alpha*self.alpha
		self.finish_calculations()

#a function for running parallel:
def run_single_sim(a):
    a.run_sim()
    #note that here we loose the functions of all classes and we just have data!
    table=a.table()
    del a
    return table


