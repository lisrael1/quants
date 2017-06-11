import numpy as np
number_of_inputs=2 #if you change here, you need to change also here a[:,-1]+=np.matrix([1,1,1]).T
max_val=3
min_val=-max_val

i=0
while True:
    i+=1
    a=np.matrix(np.random.randint(min_val,max_val,number_of_inputs**2)).reshape(number_of_inputs,number_of_inputs)
    a[:,-1]=np.matrix([-1*sum(i.A1) for i in a[:,0:-1]]).T
    ##a[:,-1]+=-np.ones((number_of_inputs,1),int)
    a[:,-1]+=np.matrix([1,-1]).T
    if np.linalg.matrix_rank(a)==number_of_inputs:
            break
    if i==1000:
        print "cannot find A matrix, exit"
        exit()
aI=a.I

print "matrix:"
print a
print "sum of rows:"
print np.matrix([sum(i.A1) for i in a]).T
print "rank:"
print np.linalg.matrix_rank(a)
print "det:"
print np.linalg.det(a)
print "a invert:"
print aI
print "a*aI:"
print (aI*a).round(10)

import pylab as pl
mean = [0, 0]
c=9
cov = [[10, c], [c, 10]]  # diagonal covariance
x = np.mat(np.random.multivariate_normal(mean, cov, 500).T)

#print a*x
#print ((a*x)[0]).A1,((a*x)[1]).A1

pl.scatter(((x)[0]).A1,((x)[1]).A1,label='before A')
pl.scatter(((a*x)[0]).A1,((a*x)[1]).A1,label='after A')
pl.legend()
pl.text(0,7,a)
pl.show()