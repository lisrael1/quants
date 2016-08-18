#!/usr/bin/python

'''
here we take random y and x, and try to run the flow on those y and x. 
we run the flow only with modulo, without dither and alpha (meaning that they are 0 and 1)
'''

from numpy import *
from numpy import matrix as m
from pylab import plot,show,legend,grid

set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=100, suppress=1, nanstr=None, infstr=None, formatter=None)

#if the modulo size for example is 1.5 so we will not change the values between -1.5 and 1.5, and 1.6 will become -1.4
def mod(num,modulo_size):
#    return multiply(m(sign(num)),m(num)%modulo_size)
    return m((m(num)+modulo_size)%(2*modulo_size)-modulo_size)
def quantizer(left,right,options):
    delta=1.0*(right-left)/options
    return r_[left+delta/2:right:delta]
def quantizise(numbers,quants):
    return m([min(quants, key=lambda x:abs(x-number)) for number in numbers.A1]).reshape(numbers.shape)

mod_size=1.5

y=random.uniform(-1.5,1.5,9).tolist()
x=m([i+0.1+random.normal(0,0.1) for i in y]).tolist()

nx=mod(x,mod_size)
ny=mod(y,mod_size)


if 0:
	q=quantizer(-mod_size,mod_size,70)
	nx=quantizise(nx,q)
	ny=quantizise(ny,q)

#A=m([[1,-1],[-2,1]])
#c=concatenate((nx,ny))
#d=c.T*A
#print A
#print d*A.I

z=mod(nx-y,mod_size)
x=m(x)
y=m(y)
if 0:
	print "mod(x-y,mod_size): ",mod(x-y,mod_size)
	print "mod(mod(x,mod_size)-mod(y,mod_size),mod_size): ",mod(x-y,mod_size)
	print "mod(x-mod(y,mod_size),mod_size): ",mod(x-mod(y,mod_size),mod_size)
	print "mod(mod(x,mod_size)-y,mod_size): ",mod(mod(x,mod_size)-y,mod_size)
	exit()

print "error:",z+y-x
print "error sum: ",m(sum((z+y-x).A1))
print "original  x: ",x
print "recovered x: ",z+y
print "original y: ",y
print "x after modulo: ",nx
print "x-y after modulo: ",nx-y
print "y after modulo: ",ny
print "z :",z



#plot(x,0,'*',label="x")
#plot(y,0,'*',label="y")
#plot(nx,0,'*',label="nx")
#plot(ny,0,'*',label="ny")
#plot(z,0,'*',label="z")
#legend()
##show()
