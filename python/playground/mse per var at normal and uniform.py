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

def calc_mse(est,real):
    e=est-real
    return e*e.T/e.size
class fl(pc):
    def __init__(self,var):
        self.s=2000000.0
        self.var=var
        self.sig_x=1.0

        self.x=mat(random.normal(0,self.sig_x,self.s))
        self.u=random.uniform(-sqrt(self.var*12)/2,sqrt(self.var*12)/2,self.s)
        self.n=random.normal(0,sqrt(self.var),self.s)

        self.mse_uniform=calc_mse(self.x+self.u,self.x)
        self.mse_normal=calc_mse(self.x+self.n,self.x)

#run sim:
d=pd.concat([fl(i).table() for i in arange(0.1,10,0.3)],axis=1).transpose()

#results:

pylab.plot(d["var"],d["mse_uniform"],label="mse_uniform")
pylab.plot(d["var"],d["mse_normal"],label="mse_normal")

pylab.xlabel('var')
pylab.ylabel('mse')
pylab.legend(loc="best")
pylab.grid()
pylab.show()
