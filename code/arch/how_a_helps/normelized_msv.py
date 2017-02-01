from numpy import matrix
from numpy.random import normal,uniform,randint,choice
import numpy as np
import pylab

def quantizer(left,right,options):
    delta=1.0*(right-left)/options
    return np.r_[left+delta/2:right:delta]
#this function quantize a number by a giver quantizer with bins
def quantizise(mumber,quants):
    return min(quants, key=lambda x:abs(x-mumber))

middle=10
q=quantizer(middle-2,middle+2,2)
i=normal(middle,40,10000)
o=[quantizise(j,q) for j in i]

print np.mean((i-o)**2)
print np.mean((i-o)**2/np.var(i))
