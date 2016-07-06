#!/usr/bin/python
from numpy import *
from numpy import matrix as m

def mo(num,mod):
    return (m(num)+mod)%(2*mod)-mod
def quantizer(left,right,options):
    delta=1.0*(right-left)/options
    return r_[left+delta/2:right:delta]
def quantizise(numbers,quants):
    return m([min(quants, key=lambda x:abs(x-number)) for number in numbers.A1]).reshape(numbers.shape)

y=random.uniform(-100,100,9).tolist()
x=m([i+0.1+random.normal(0,1) for i in y]).tolist()

mod_size=8
nx=mo(x,mod_size)
ny=mo(y,mod_size)
print "x after modulo: ",(nx).tolist()
print "y after modulo: ",(ny).tolist()
if 0:
	q=quantizer(-mod_size,mod_size,70)
	nx=quantizise(nx,q)
	ny=quantizise(ny,q)

#A=m([[1,-1],[-2,1]])
#c=concatenate((nx,ny))
#d=c.T*A
#print A
#print d*A.I

z=mod(nx-ny,mod_size)

print z+y-x
print sum((z+y-x).A1)
print "recovered x: ",(z+y).tolist()
print "original x: ",x
print "original y: ",y

