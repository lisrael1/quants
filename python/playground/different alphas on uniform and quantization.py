##comparing the variance of quantization vs var of uniform dist at the same size
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
class fl(pc):
    def __init__(self,alpha):
        self.alpha=alpha
        self.s=2000000.0
##        self.s=1.0
        self.bin_size=3.1
        self.sig_x=1.0

        self.x=mat(random.normal(0,self.sig_x,self.s))
        self.u=mat(random.uniform(-self.bin_size/2,self.bin_size/2,self.s))

        self.x_with_u=self.x+self.u

        self.x_qnt=qnt(self.bin_size,self.x_with_u)
        self.x_qnt_less_u=self.x_qnt-self.u
        self.x_qnt_less_u_with_alpha=self.x_qnt_less_u*self.alpha

        self.e=self.x_qnt_less_u_with_alpha-self.x
##        print self.e
        self.mse=self.e*self.e.T/self.s

        self.e_u=self.x_with_u*self.alpha-self.x
        self.mse_u=self.e_u*self.e_u.T/self.s
##        print self.mse
##        self.x_qnt_var=var(self.x_qnt_less_u_with_alpha)


#run sim:
d=pd.concat([fl(i).table() for i in arange(0,1.5,0.1)],axis=1).transpose()

#results:
if 0:
    print d
else:
    pylab.plot(d["alpha"],d["mse"],label="mse")
    pylab.plot(d["alpha"],d["mse_u"],label="mse_u")
    pylab.xlabel('alpha')
    pylab.ylabel('mse')
    pylab.grid()
    pylab.legend(loc="best")
    pylab.show()
