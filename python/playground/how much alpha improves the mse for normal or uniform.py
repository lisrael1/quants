from numpy import *
import pandas as pd
from collections import OrderedDict #you dont need this. only that the varialbes will be at their original order...

class pc():
	  def  __iter__(self):
	  	   return self.__dict__.iteritems()
	  def table(self):
			data=pd.DataFrame([{k:v for k,v in dict(self).iteritems() if t(v)}])
        	#return data
			return data.transpose()
	  def t(v):
		try:
			return size(v)<10
		except:
			   return False
def qnt(bin_size,numbers):
	ret=bin_size*rint(numbers/bin_size)
	return ret
class fl(pc):
    def __init__(self,sig_z):
        self.noise_type="normal"
##        self.noise_type="u" #if you choose not normal, put at the arange up to 7
        self.s=2000000.0
        self.bin_size=1.0
        self.sig_x=1.0
        self.sig_z=sig_z
        self.x=mat(random.normal(0,self.sig_x,self.s))
        if self.noise_type=="normal":
            self.z=mat(random.normal(0,self.sig_z,self.s))
        else:
            self.z=mat(random.uniform(-self.sig_z/2,self.sig_z/2,self.s))
##        self.z=mat(random.uniform(-self.sig_z/2,self.sig_z/2,self.s))
        self.new=self.x+self.z

        if self.noise_type=="normal":
            self.a=self.sig_x**2/(self.sig_x**2+self.sig_z**2)
        else:
            self.a=self.sig_x**2/(self.sig_x**2+self.sig_z**2/12)


        self.mse=self.new-self.x
        self.mse_with_a=self.a*self.new-self.x
        self.mse=self.mse*self.mse.T/self.s
        self.mse_with_a=self.mse_with_a*self.mse_with_a.T/self.s

        self.var_new=var(self.new)
        self.var_x=var(self.x)
        self.var_z=var(self.z)
        self.mse_with_a_div_mse=self.mse_with_a/self.mse



#run sim:
d=pd.concat([fl(i).table() for i in arange(0.1,1.5,0.05)],axis=1).transpose()

#results:
##print d
pylab.plot(d["sig_z"],d["mse_with_a"],label="mse_with_a")
pylab.plot(d["sig_z"],d["mse"],label="mse")
pylab.plot(d["sig_z"],d["mse_with_a_div_mse"],label="mse_with_a_div_mse")
pylab.xlabel('sig_z')
pylab.ylabel('MSE')

pylab.legend(loc="best")
pylab.grid()
pylab.show()
