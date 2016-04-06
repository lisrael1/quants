from random import randint as rand
from numpy import sign
import numpy as np
import pylab


print_itter=0



#those values are from mu=0, sigma=1, modulo=2, and we have 13 bins
bars=[-1.4658,-1.1993,-0.9328,-0.6663,-0.3998,-0.1333,0.1333,0.3998,0.6663,0.9328,1.1993,1.4658]
vals=[-1.8032,-1.3325,-1.0660,-0.7995,-0.5330,-0.2665,0.0000,0.2665,0.5330,0.7995,1.0660,1.3325,1.8032]

def get_quantized(num):
	c=bars+[num]
	c.sort()
	return vals[c.index(num)]


from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()


def get_err(modulo,rounds):
	e=[]
	for i in range (rounds):
		num=rand(-2e3,2e3)/1.0e3
		num=np.random.normal(0,1,1)
		#qu=get_quantized(num)
		#err_wo_m= abs(qu-num)
		#if print_itter:
		#	print "random:",num
		#	print "quantized is :",qu
		#	print "error is: ",err_wo_m

		#doing modulo (also for negative modulo):
		num_mod=sign(num)*(abs(num)%modulo)
		qu=get_quantized(num_mod)
		err_w_m= abs(qu-num)
		if print_itter:
			print "number with modulo: ",num_mod
			print "quantized is :",qu
			print "error is: ",err_w_m

		#delta=abs(err_w_m-err_wo_m)
		#e+=[delta]
		#if print_itter:
		#	print "delta is: ",delta
		#	print
		e+=[err_w_m]
		if print_itter:
			print "error is: ",err_w_m
	e=sum(e)/rounds
	return e


if print_itter:
	print "bars:\t\t"+"\t\t".join(str(i) for i in bars)
	print "vals:"+"\t\t".join(str(i) for i in vals)


m=[]
er=[]
rounds=10000

for modulo in np.r_[0.01:4:0.01]:
	e=get_err(modulo,rounds)
	print "average error for modulo",modulo,"\tis",e
	m+=[modulo]
	er+=[e]
pylab.plot(m,er)
pylab.show()

#another way to use parallel:
#results = Parallel(n_jobs=num_cores)(delayed(get_err)(modulo,100000) for modulo in np.r_[0.01:2.5:0.01]) 
#print results
#exit()



