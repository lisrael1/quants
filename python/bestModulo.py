import numpy as np
from pylab import *
import itertools

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

###functions###
#modolu around 0. we will right shift (+mod) it between 0 and 2*modulo, do modolu and left shift it by -mod
def modOp(num,mod):
    return (num+mod)%(2*mod)-mod
modOpVec=np.vectorize(modOp)

#if we have left and right border and we want to put quantizer with #options
def quantizer(left,right,options):
    delta=1.0*(right-left)/options
    return np.r_[left+delta/2:right:delta]
def quantizise(mumber,quants):
    min(quants, key=lambda x:abs(x-mumber))
quantiziseVec=np.vectorize(quantizise)
absVec=np.vectorize(abs)
###end of functions###


#2e5 is very sharp, 2e4 is sharp, 2e3 has a lot of edges. you can use 5e3
samples=4000
ditherOn=1


#given modulo and number of bins
def getError(modulo,numOfBins):
	ditherSize=2.0*(modulo)/numOfBins
	quants=quantizer(-modulo,modulo,numOfBins)
	#randomizing numbers and dithers
	originalNum=np.matrix(np.random.normal(0,1,samples))
	#originalNum=np.matrix(np.random.uniform(-1,1,samples))
	if ditherOn:
		dither=np.matrix(np.random.uniform(-ditherSize/2,ditherSize/2,samples))
	else:
		dither=0
	#adding dither:
	newNum=originalNum+dither
	#modulo:
	newNum=modOpVec(newNum,modulo)
	#quantizer:
	#newNum=quantiziseVec(newNum,quants)
	newNum=np.matrix([min(quants, key=lambda x:abs(x-i)) for i in newNum.tolist()[0]])
	#newNum=map(lambda x: quantizise(x,quants),newNum)
	#subtract the dither:
	newNum-=dither
	if ditherOn:
		newNum*=1/(1+1/9)
	#modulo:
	newNum=modOpVec(newNum,modulo)
	error=absVec(newNum-originalNum)
	error=error.tolist()[0]
	error=sum(error)/len(error)
	return [error,modulo,numOfBins]
getErrorVec=np.vectorize(getError)


modulo=np.r_[0.5:4:0.1].tolist()
numOfBins=range(2,40)
errors=Parallel(n_jobs=num_cores)(delayed(getErrorVec)(i[0],i[1]) for i in list(itertools.product(modulo,numOfBins)))
errors=np.matrix(errors)

plot(errors)
show()
