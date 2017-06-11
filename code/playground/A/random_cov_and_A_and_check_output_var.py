import numpy as np
import pylab as pl
number_of_inputs=2 #if you change here, you need to change also here a[:,-1]+=np.matrix([1,1,1]).T
max_val=3
min_val=-max_val

def get_A():
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
##    print "sum of rows:"
##    print np.matrix([sum(i.A1) for i in a]).T
##    print "rank:"
##    print np.linalg.matrix_rank(a)
##    print "det:"
##    print np.linalg.det(a)
##    print "a invert:"
##    print aI
##    print "a*aI:"
##    print (aI*a).round(10)
##    print "eigenvalues:",np.linalg.eig(a)[0]

    return a



def get_cov():
    varx=np.random.uniform(0,10)
    vary=np.random.uniform(0,10)
    c_limit=np.sqrt(varx*vary)
    c=np.random.uniform(-c_limit,c_limit)
    cov = np.matrix([[varx, c], [c, vary]])

    print "cov:"
    print cov
    return cov



##exit()


def print_x(a,cov):
    mean = [0, 0]
    x = np.mat(np.random.multivariate_normal(mean, cov, 50000).T)

    #print a*x
    #print ((a*x)[0]).A1,((a*x)[1]).A1


    pl.scatter(((x)[0]).A1,((x)[1]).A1,label='before A')
    pl.scatter(((a*x)[0]).A1,((a*x)[1]).A1,label='after A')
    pl.legend()
    prnt=a.tolist()+cov.tolist()
    pl.text(0,7,prnt)

    print
    print "new cov:"
    print np.cov((a*x), bias=1)

for i in range(10):
    a=get_A()
    cov=get_cov()
    print
    print "var x=",cov[0,0]
    print "var y=",cov[1,1]

    v_x_new=cov[0,0]*a[0,0]*a[0,0]   +cov[1,1]*a[0,1]*a[0,1]   +2*a[0,0]*a[0,1]*cov[0,1]
    v_y_new=cov[0,0]*a[1,0]*a[1,0]   +cov[1,1]*a[1,1]*a[1,1]   +2*a[1,0]*a[1,1]*cov[0,1]
    print "var x new=",v_x_new
    print "var y new=",v_y_new

    print
    print "sig x=",np.sqrt(cov[0,0])
    print "sig y=",np.sqrt(cov[1,1])
    print "sig x new=",np.sqrt(v_x_new)
    print "sig y new=",np.sqrt(v_y_new)
    print
    print "x strach=",np.sqrt(v_x_new)/np.sqrt(cov[0,0])
    print "y strach=",np.sqrt(v_y_new)/np.sqrt(cov[1,1])
    print "eigenvalues:",np.linalg.eig(a)[0]
    print "det:",np.linalg.det(a)

    if 1:#if you want to see the A input output, to see the strach
        print_x(a,cov)
        pl.show()