import pandas as pd
import pylab as plt
import numpy as np
import inspect




class flow():
    def __init__(self,init_seed,init_bin_size,init_samples,init_modulo_size,init_quantizer_offset,init_cov,init_A):
        for varname in inspect.currentframe().f_code.co_varnames[1:]:#[1:] to remove self
            setattr(self, varname, locals()[varname])
            z=100
	    self.init_cov=np.mat(init_cov)
	    self.init_A=np.mat(init_A)


    def __iter__(self):#just for dict function
		return self.__dict__.iteritems()
    def table(self):
        data=pd.DataFrame([dict(self)]).transpose()
        return data

    def mod(self,numbers,modulo_size_edge_to_edge):
        if modulo_size_edge_to_edge==0:#TODO just remove column and return it when finishing
            return numbers
        '''
            modulo_size_edge_to_edge should be array of size 1 or at numbers column size, for example:
                mod(numbers,[6.0]
                mod(numbers,[6.0,100])#for mod 6 for the first column and 100 for the second column
        '''
        modulo_size_edge_to_edge=np.mat(modulo_size_edge_to_edge)
##        return (numbers+modulo_size_edge_to_edge/2.0)%(modulo_size_edge_to_edge)-modulo_size_edge_to_edge/2.0
        num_resize=(numbers+modulo_size_edge_to_edge/2.0)%modulo_size_edge_to_edge
        num_resize-=(modulo_size_edge_to_edge/2.0)
        return num_resize
    def quantize(self,numbers,offset,bin_size):
        if bin_size==0:#TODO just remove column and return it when finishing
            return numbers
        '''
            offset and bin_size should be array of size 1 or at numbers column size, for example:
                quantize(numbers,[0],[0.1]
                quantize(numbers,[0.05,0.125],[0.1,0.25])
        '''
        dim=numbers.shape[1]
        quantizer=np.mat(np.identity(dim)*bin_size)
        return ((numbers+offset)*quantizer.I).round()*quantizer-offset

    def generate_data(self,cov,number_of_samples):
        cov=np.mat(cov)
        dim=cov.shape[1]
        data=np.mat(np.random.multivariate_normal(np.zeros(dim), cov, int(number_of_samples)))
        return data



    def run(self):
        self.flow_a_data=self.generate_data(self.init_cov,self.init_samples)
        self.flow_b_enc_mod=self.mod(self.flow_a_data,self.init_modulo_size)
        self.flow_c_enc_quantized=self.quantize(self.flow_b_enc_mod,self.init_quantizer_offset,self.init_bin_size)
        self.flow_d_dec_after_a=self.flow_c_enc_quantized*self.init_A.T
        self.flow_e_dec_mod=self.mod(self.flow_d_dec_after_a,self.init_modulo_size)
        self.flow_f_dec_inv_a=self.flow_e_dec_mod*self.init_A.I.T
	self.flow_h_output=self.flow_f_dec_inv_a
        self.flow_g_error=np.mat((self.flow_f_dec_inv_a-self.flow_a_data).round(2))
        self.mse_error=self.flow_g_error.flatten()*self.flow_g_error.flatten().T/self.flow_g_error.size
        print self.mse_error




init_seed=np.random.randint(1,1000)
##init_seed=475
print init_seed
np.random.seed(init_seed)

#at 8G ram pc put max 1e6. you can run 1e7 but not collect more than 5 results. 
init_samples=1e6
init_bin_size=[0.05,0.00001]
init_bin_size=[0.15]
init_modulo_size=[5,1200]
init_modulo_size=[5]
_cov=9.99999

init_quantizer_offset=[0]
init_cov=[[10,_cov],[_cov,10]]
init_A=[[1,-1],[0,1]]

delta=pd.DataFrame()
if 1:
	for _cov in [5,8,8.5,9,9.5,9.7,9.9,9.95,9.9999,9.99999,9.999999,10]:
		init_cov=[[10,_cov],[_cov,10]]
		f=flow(init_seed,init_bin_size,init_samples,init_modulo_size,init_quantizer_offset,init_cov,init_A)
		f.run()
		delta=delta.append(pd.DataFrame(f.flow_h_output[:,0]-f.flow_h_output[:,1],columns=[f.init_cov[1,0]]))
	delta.hist(bins=1000)
	#plt.suptitle("var(x)=var(y)=10,y w/o quantizer and modulo, x with 0.05 qunatizer and 10 mod. hist of x-y, per cov:")
	plt.suptitle("var(x)=var(y)=10,x and y with 0.15 qunatizer and 10 mod. hist of x-y, per cov:")
	plt.show()
##    plt.scatter(f._a_data[:,0],f._a_data[:,1])
##    plt.scatter(f._e_dec_mod[:,0],f._e_dec_mod[:,1])
##    plt.show()
else:#put only 1 sample per sim
    for i in range(10000):
        f=flow()
        f.run()
        if f.mse_error!=0:
            break
#f.table().to_excel('hi.xlsx')
