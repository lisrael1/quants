from numpy import *

s=2000000.0
sig_x=100.0
sig_z=70.0
x=mat(random.normal(0,sig_x,s))
z=mat(random.normal(0,sig_z,s))
new=x+z

a=sig_x**2/(sig_x**2+sig_z**2)


e1=new-x
e2=a*new-x
e1=sqrt(e1*e1.T/s)
e2=sqrt(e2*e2.T/s)

print a

print "vars"
print var(new)
print var(x)
print var(z)
print var(x)/var(new)

print "errors"
print e1
print e2
print e2/e1