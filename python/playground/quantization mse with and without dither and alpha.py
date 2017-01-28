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
    def __init__(self,alpha):
        self.alpha=alpha
        self.s=2000000.0
##        self.s=1.0
        self.bin_size=0.2
        self.sig_x=1.0

        self.x=mat(random.normal(0,self.sig_x,self.s))

        self.x_qnt_without_dither=qnt(self.bin_size,self.x)
        self.mse_wo_dither=calc_mse(self.x_qnt_without_dither,self.x)

        self.u=mat(random.uniform(-self.bin_size/2,self.bin_size/2,self.s))
        self.x_with_u=self.x+self.u
        self.x_qnt_with_dither=qnt(self.bin_size,self.x_with_u)-self.u
        self.mse_w_dither=calc_mse(self.x_qnt_with_dither*alpha,self.x)


        self.ratio=self.mse_w_dither/self.mse_wo_dither





#run sim:
d=pd.concat([fl(i).table() for i in arange(0.8,1.2,0.01)],axis=1).transpose()

#results:
if 0:
    print d
else:
    pylab.plot(d["alpha"],d["ratio"],label="ratio")
##    pylab.plot(d["bin_size"],d["mse_wo_dither"],label="mse_wo_dither")
    pylab.xlabel('alpha')
    pylab.ylabel('mse')
    pylab.grid()
    pylab.legend(loc="best")
    pylab.show()
