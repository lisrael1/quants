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
    def __init__(self,try_alpha):
        self.s=2000000.0
        self.bin_size=1.0
        self.sig_x=1.0
        self.sig_z=0.7
        self.x=mat(random.normal(0,self.sig_x,self.s))
        self.z=mat(random.normal(0,self.sig_z,self.s))
##        self.z=mat(random.uniform(-self.sig_z/2,self.sig_z/2,self.s))
        self.new=self.x+self.z


        self.try_alpha=try_alpha
        self.a=self.sig_x**2/(self.sig_x**2+self.sig_z**2)


        self.e=self.new-self.x
        self.e_with_a=self.try_alpha*self.new-self.x
        self.e=self.e*self.e.T/self.s
        self.e_with_a=self.e_with_a*self.e_with_a.T/self.s

        self.var_new=var(self.new)
        self.var_x=var(self.x)
        self.var_z=var(self.z)
        self.e_with_a_div_e=self.e_with_a/self.e



#run sim:
d=pd.concat([fl(i).table() for i in arange(0,1.5,0.05)],axis=1).transpose()

#results:
##print d
pylab.plot(d["try_alpha"],d["e_with_a"],label="e_with_a")
pylab.xlabel('try_alpha')
pylab.ylabel('MSE')
text="best calculated a is "+str(d["a"].iloc[0])
pylab.text(d["a"].iloc[0],0.90,text)
##pylab.legend(loc="best")
pylab.grid()
pylab.show()

print d["var_new"].iloc[0]
print d[["try_alpha","e_with_a"]]
