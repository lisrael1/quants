from numpy import *
import pandas as pd
import pylab

class pc():
	  def  __iter__(self):
	  	   return self.__dict__.iteritems()
	  def table(self):
			data=pd.DataFrame([{k:v for k,v in dict(self).iteritems() if self.t(v)}])
        	#return data
			return data.transpose()
	  def t(self,v):
		try:
			return size(v)<10
		except:
			   return False
def qnt(bin_size,numbers):
	ret=bin_size*rint(numbers/bin_size)
	return ret
def calc_mse(est,real):
    e=est-real
    return e*e.T/e.size
class fl(pc):
    def __init__(self,bin_size):
        self.s=2000000.0
        self.bin_size=bin_size
        self.sig_x=1.0

        self.x=mat(random.normal(0,self.sig_x,self.s))
        self.u=random.uniform(-self.bin_size/2,self.bin_size/2,self.s)
        self.x_qnt=qnt(self.bin_size,self.x)

        self.x_qnt_var=var(self.x_qnt)
        self.x_var=var(self.x)
        self.delta_var=self.x_var-self.x_qnt_var
        self.uniform_var_for_bin_size=self.bin_size**2/12+self.x_var
        self.uniform_var_for_bin_size_sim=var(self.x+self.u)
##        self.uniform_var_for_bin_size_sim=var(self.x+random.normal(0,self.bin_size,self.s))
##        self.uniform_quant_delta=self.x_qnt_var-self.uniform_var_for_bin_size
        self.mse_uniform=calc_mse(self.x+self.u,self.x)
        self.mse_quant=calc_mse(self.x_qnt,self.x)

#run sim:
d=pd.concat([fl(i).table() for i in arange(0.1,10,0.3)],axis=1).transpose()

#results:
##print d
pylab.subplot(2, 1, 1)
pylab.plot(d["bin_size"],d["x_qnt_var"],label="x_qnt_var")
pylab.plot(d["bin_size"],d["uniform_var_for_bin_size"],label="uniform_var_for_bin_size")
##pylab.plot(d["bin_size"],d["uniform_quant_delta"],label="uniform_quant_delta")
pylab.plot(d["bin_size"],d["uniform_var_for_bin_size_sim"],label="uniform_var_for_bin_size_sim")

pylab.xlabel('bin_size')
pylab.ylabel('var')
pylab.legend(loc="best")
pylab.grid()

pylab.subplot(2, 1, 2)
pylab.plot(d["bin_size"],d["mse_uniform"],label="mse_uniform")
pylab.plot(d["bin_size"],d["mse_quant"],label="mse_quant")
pylab.xlabel('bin_size')
pylab.ylabel('mse')

pylab.legend(loc="best")
pylab.grid()
pylab.show()
