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
best_bin_sizes_at_1_var={1: 0, 2: 0.0616666, 3: 1.64, 4: 1.342, 5: 1.186666, 6: 1.0021428571428579, 7: 0.91375, 8: 0.838888, 9: 0.7565, 10: 0.660454, 11: 0.6333333, 12: 0.57346153846153891, 13: 0.535, 14: 0.5106666, 15: 0.4821875, 16: 0.4529411764705884, 17: 0.434166666, 18: 0.415263157894, 19: 0.38925, 20: 0.36142857142857165, 21: 0.368181818181, 22: 0.34456521739130475, 23: 0.32666666, 24: 0.3184, 25: 0.30692307692307708, 26: 0.28555555555, 27: 0.28392857142857153, 28: 0.27741379310344838, 29: 0.265833333, 30: 0.25983870967741973, 31: 0.245625, 32: 0.2360606060606063, 33: 0.23573529411764715, 34: 0.21785714285714297, 35: 0.22208333333333341, 36: 0.21310810810810832, 37: 0.21381578947368451, 38: 0.20243589743589752, 39: 0.188}
class fl(pc):
    def __init__(self,bin_size,number_of_bins):
##        print bin_size
        self.bin_size=bin_size
        self.number_of_bins=number_of_bins
        self.var_x=1.0

        self.quantizatio_var=self.bin_size**2/12

        self.new_var=self.var_x+self.quantizatio_var

        self.fix=self.quantizatio_var/self.new_var
        self.fix_var=self.var_x+1/(1+self.var_x/self.quantizatio_var)






#run sim:

##d=pd.concat([fl(i).table() for i in arange(0.26,1.7,0.01)],axis=1).transpose().reset_index()



d=pd.concat([fl(best_bin_sizes_at_1_var[i],i).table() for i in range(3,25)],axis=1).transpose().reset_index()

pylab.subplot(3,1,1)
pylab.xticks(range(0,40))
pylab.yticks(linspace(0,0.2,15))
pylab.plot(d["number_of_bins"],d["fix"],label="fix")
pylab.xlabel('number_of_bins')
pylab.ylabel('fix')
pylab.grid()
##pylab.legend(loc="best")

pylab.subplot(3,1,3)
pylab.xticks(range(0,40))
pylab.yticks(linspace(1,1.25,15))
##pylab.subplot(3,1,3).set_xticks(range(0,40))
##pylab.subplot(3,1,3).set_yticks(linspace(0,0.2,15))
pylab.plot(d["number_of_bins"],d["new_var"],label="new_var")
pylab.plot(d["number_of_bins"],d["fix_var"],label="fix_var")
pylab.xlabel('number_of_bins')
pylab.ylabel('mse')
pylab.legend(loc="best")
pylab.grid()

d=pd.concat([fl(i,0).table() for i in arange(0.26,1.7,0.01)],axis=1).transpose().reset_index()

pylab.subplot(3,1,2)
pylab.xticks(linspace(0,1.7,30))
pylab.yticks(linspace(0,0.2,15))
pylab.plot(d["bin_size"],d["fix"],label="fix")
pylab.xlabel('bin_size')
pylab.ylabel('fix')
pylab.grid()
##pylab.legend(loc="best")
pylab.show()
