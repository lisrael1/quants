import pandas as pd
import pylab as plt
import numpy as np
import inspect



 ####  # #    #    ###### #    # #    #  ####  ##### #  ####  #    #  ####  
#      # ##  ##    #      #    # ##   # #    #   #   # #    # ##   # #      
 ####  # # ## #    #####  #    # # #  # #        #   # #    # # #  #  ####  
     # # #    #    #      #    # #  # # #        #   # #    # #  # #      # 
#    # # #    #    #      #    # #   ## #    #   #   # #    # #   ## #    # 
 ####  # #    #    #       ####  #    #  ####    #   #  ####  #    #  ####  

class flow():
	'''
		this class is for running full system flow of:
			X->mod->quantize->A->mod->inv(A)
		inputs are covariance matrix, number of samples, A, mod and quantizer
	'''
	def __init__(self,init_seed,init_bin_size,init_samples,init_modulo_size,init_quantizer_offset,init_cov,init_A):
		'''
			this function is for taking the class arguments...
		'''
		for varname in inspect.currentframe().f_code.co_varnames[1:]:#[1:] to remove self
			setattr(self, varname, locals()[varname])
		self.init_cov=np.mat(self.init_cov)
		self.init_A=np.mat(self.init_A)
		self.init_modulo_size=np.mat(self.init_modulo_size)
		self.run()

	def generate_data(self,cov,number_of_samples):
		dim=cov.shape[1]
		#multivariate_normal(mean,cov,samples):
		data=np.mat(np.random.multivariate_normal(np.zeros(dim), cov, int(number_of_samples)))
		return data
	
	def mod(self,numbers,modulo_size_edge_to_edge):
		'''
			modulo_size_edge_to_edge should be array of size 1 or at numbers column size, for example:
				mod(numbers,[6.0]
				mod(numbers,[6.0,100])#for mod 6 for the first column and 100 for the second column
			modulo_size_edge_to_edge should be numpy matrix
			TODO add modulo size 0 for disabling modulo - just remove column and return it when finishing
		'''
		num_resize=(numbers+modulo_size_edge_to_edge/2.0)%modulo_size_edge_to_edge
		num_resize-=modulo_size_edge_to_edge/2.0
		return num_resize
	def quantize(self,numbers,offset,bin_size):
		'''
			offset and bin_size should be array of size 1 or at numbers column size, for example:
				quantize(numbers,[0],[0.1]
				quantize(numbers,[0.05,0.125],[0.1,0.25])
			TODO add quantizer size 0 for disabling - just remove column and return it when finishing
		'''
		dim=numbers.shape[1]
		quantizer=np.mat(np.identity(dim)*bin_size)
		return ((numbers+offset)*quantizer.I).round()*quantizer-offset




	def run(self):
		#flow: X->mod->quantize->A->mod->inv(A)
		self.flow_a_data=self.generate_data(self.init_cov,self.init_samples)
		self.flow_b_enc_mod=self.mod(self.flow_a_data,self.init_modulo_size)
		self.flow_c_enc_quantized=self.quantize(self.flow_b_enc_mod,self.init_quantizer_offset,self.init_bin_size)
		self.flow_d_dec_after_a=self.flow_c_enc_quantized*self.init_A.T
		self.flow_e_dec_mod=self.mod(self.flow_d_dec_after_a,self.init_modulo_size)
		self.flow_f_dec_inv_a=self.flow_e_dec_mod*self.init_A.I.T
		self.flow_h_output=self.flow_f_dec_inv_a
		self.flow_g_error=(self.flow_h_output-self.flow_h_output).round(2)
		self.mse_error=self.flow_g_error.flatten()*self.flow_g_error.flatten().T/self.flow_g_error.size


	#next 2 functions are just for dumping all class element into excel
	def __iter__(self):#just for dict function 
		return self.__dict__.iteritems()
	def table(self):
		return pd.DataFrame([dict(self)]).transpose()









                                        
#####  #    # #    #     ####  # #    # 
#    # #    # ##   #    #      # ##  ## 
#    # #    # # #  #     ####  # # ## # 
#####  #    # #  # #         # # #    # 
#   #  #    # #   ##    #    # # #    # 
#    #  ####  #    #     ####  # #    # 



#generating simulation seed to you can recover spesicif results:
init_seed=np.random.randint(1,1000) if True else 473
np.random.seed(init_seed)

#at 8G ram pc put max 1e6. you can run 1e7 but not collect more than 5 results. 
init_samples=4

#you can have here different bin size per data column:
init_bin_size=[0.05,0.00001]
init_modulo_size=[5,1200]
#running over last config and putting the same bin size and modulo for all data columns
init_bin_size=[0.15]
init_modulo_size=[5]

_cov=9.99999
init_quantizer_offset=[0]
init_cov=[[10,_cov],[_cov,10]]
init_A=[[1,-1],[0,1]]

delta=pd.DataFrame()
if 1:
	for _cov in [5,8,8.5,9,9.5,9.7,9.9,9.95,9.9999,9.99999,9.999999,10]:#trying different cov and plotting them
		init_cov=[[10,_cov],[_cov,10]]
		#running sim:
		f=flow(init_seed,init_bin_size,init_samples,init_modulo_size,init_quantizer_offset,init_cov,init_A)
		#taging only what is interesting from the last sim:
		print f.table()
		exit()
		delta=delta.append(pd.DataFrame(f.flow_h_output[:,0]-f.flow_h_output[:,1],columns=[f.init_cov[1,0]]))
        	#if f.mse_error!=0:
	delta.hist(bins=1000)
	#plt.suptitle("var(x)=var(y)=10,y w/o quantizer and modulo, x with 0.05 qunatizer and 10 mod. hist of x-y, per cov:")
	plt.suptitle("var(x)=var(y)=10,x and y with 0.15 qunatizer and 10 mod. hist of x-y, per cov:")
	plt.show()
##    plt.scatter(f._a_data[:,0],f._a_data[:,1])
##    plt.scatter(f._e_dec_mod[:,0],f._e_dec_mod[:,1])
##    plt.show()
#f.table().to_excel('hi.xlsx')
