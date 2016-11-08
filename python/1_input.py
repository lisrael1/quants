#!/usr/bin/python
import sys
sys.path.append('./functions')
from functions import *

"""
i can run this by rewriting the generate data to 
what i want to see here is how the mse at with and without modulo
so need to run range of bins and 'modulo' and find the minimum for each bin and run it with and without modulo

we can split table by specific values, and put for each a curve
then we have to sort by x,y, commonly x is number of bins
"""

a=[]
d=[data_multi_inputs(
	number_of_inputs=1,
	number_of_samples=4e1,#dont put above 4e5
	var=0,
	covar=1,
	mod_size=i,
	num_quants=j,
	dither_on=d_o,
	modulo_on=m_o
	#) for i in [5] for j in [4]]
	) for i in arange(0.1,10,.05) for j in range(1,20) for m_o in [1,0] for d_o in [1,0]]
if 0:
	d=Pool().imap_unordered(n,d)
else:
	d=map(n,d)
#[i.print_all() for i in d]
#exit()
o1=[[i.mod_size,i.mse_per_input_sample,i.num_quants,i.modulo_on,i.dither_on] for i in d]
for m_o in [1,0]:
	for d_o in [1,0]:
		o=m([i for i in o1 if i[3]==m_o and i[4]==d_o])
		o=o[:,:3].tolist()
		o=lowest_y_per_x(o,2,1)
		if (0):#if you want to get the ratio between 2 options
			if (len(a)):
				b=m(o)[:,1].A1
				plot([i/j for i,j in zip(a,b)])
			a=m(o)[:,1].A1
		plot(o[:,2],o[:,1],'.--',label="m_o="+str(m_o)+",d_o="+str(d_o))
	#	for xy in zip(o[:,2].A1,o[:,1].A1):
	#		annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

xlabel("bins")
ylabel("mse")
title("best mse per bins, 1 input w/w-o modulo with ratio, var=1, dither on")
grid()
legend(loc="best")
print "simulation time: ",time() - start_time,"sec"

show()
exit()

savefig("w wo modulo at 1 input with ratio.jpg")
close()

