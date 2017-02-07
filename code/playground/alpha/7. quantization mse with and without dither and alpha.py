from numpy import *
import pandas as pd
import pylab
set_printoptions(precision=5)


min_alpha=0.74
min_alpha=0.2
max_alpha=1.04
##max_alpha=2
samples=4e6
samples=2e4

bin_size=1.6
##bin_size=0.5
##bin_size=3



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
        self.s=samples

##        self.s=1.0
        self.bin_size=bin_size
        self.var_x=1.0

        self.x=mat(random.normal(0,sqrt(self.var_x),self.s))
        self.u=mat(random.uniform(-self.bin_size/2,self.bin_size/2,self.s))
##        self.u=0

        self.x_qnt_without_dither=qnt(self.bin_size,self.x)

        self.x_with_u=self.x+self.u
        self.x_qnt_with_dither=qnt(self.bin_size,self.x_with_u)-self.u

        self.mse_wo_dither_wo_alpha=calc_mse(self.x_qnt_without_dither,self.x)
        self.mse_wo_dither_w_alpha=calc_mse(self.x_qnt_without_dither*self.alpha,self.x)
        self.mse_w_dither_w_alpha=calc_mse(self.x_qnt_with_dither*self.alpha,self.x)


        self.ratio_w_wo_dither_alpha=self.mse_w_dither_w_alpha/self.mse_wo_dither_wo_alpha
        self.ratio_dither_w_alpha=self.mse_w_dither_w_alpha/self.mse_wo_dither_w_alpha





#run sim:
d=pd.concat([fl(i).table() for i in arange(min_alpha,max_alpha,0.01)],axis=1).transpose().reset_index()

#results:
def on_subplot():
    pylab.xlabel('alpha')
    pylab.grid()
    pylab.legend(loc="best")
    pylab.xticks(arange(min_alpha,max_alpha,0.02),rotation=90)

pylab.title("bin size = "+str(bin_size))

pylab.subplot(3,1,1)
pylab.plot(d["alpha"],d["ratio_w_wo_dither_alpha"],label="ratio_w_wo_dither_alpha")
min_ratio=d.iloc[d['ratio_w_wo_dither_alpha'].idxmin()]
pylab.text(min_alpha,min_ratio['ratio_w_wo_dither_alpha'],"best ratio_w_wo_dither_alpha "+str(min_ratio['ratio_w_wo_dither_alpha'].A1[0])+" or "+str((1-min_ratio['ratio_w_wo_dither_alpha'].A1[0])*100.0)+"% at "+str(min_ratio['alpha']))
on_subplot()
pylab.ylabel('ratio_w_wo_dither_alpha')

pylab.subplot(3,1,2)
pylab.plot(d["alpha"],d["ratio_dither_w_alpha"],label="ratio_dither_w_alpha")
on_subplot()
pylab.ylabel('ratio')

pylab.subplot(3,1,3)
pylab.plot(d["alpha"],d["mse_w_dither_w_alpha"],label="mse_w_dither_w_alpha")
pylab.plot(d["alpha"],d["mse_wo_dither_w_alpha"],label="mse_wo_dither_w_alpha")
pylab.plot(d["alpha"],d["mse_wo_dither_wo_alpha"],label="mse_wo_dither_wo_alpha")
on_subplot()
pylab.ylabel('mse')


pylab.show()
