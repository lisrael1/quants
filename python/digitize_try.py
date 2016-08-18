from numpy import *



m=matrix([1.09,5.22,-3.11,0.099,0.111,-0.211]).reshape(3,2)
#q=0.1
print m
m=rint(m/0.2)*0.2

m=mat([0.8 if i>0.8 else -0.8 if i<-0.8 else i for i in m.A1]).reshape(m.shape)

#m=mat([0.8 if i>0.8 else i for i in m.A1]).reshape(m.shape)
#m=mat([-0.8 if i<-0.8 else i for i in m.A1]).reshape(m.shape)


print m

exit()

q=[-1,0,1]
i=digitize(m,q,1)
print m
print q
print m[i]
print i
