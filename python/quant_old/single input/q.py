draft!!
#!/usr/bin/python
#TODO:
#add % of error beside the mse because at range of -50 to 50 you will have more mse than -10 to 10
#main function will get the random numbers instead of generating them



#controls
searchAllRange=1     #instead of searching for minimum point - not maintain anymore!!!! use only 1
ditherOn=1          #adding uniform dither to the input
sigma=1.0
mu=0
moduloOn=1             #doing modulo
mseInsteadOfSigma=0 #seems like it's the same mse but it's faster to use dither sigma so set this to 0
samples=2e1        #2e5 is very smooth (but it will take 15 minutes), 2e4 is smooth, 2e3 has a lot of edges. 2e2 is practical for instance view. you can use 4e3
debug_before_after_dither=[]




import numpy as np
from pylab import *
import itertools
from scipy.optimize import fmin,brent
from scipy.stats import norm
from random import randint
#run parallel:
from sys import platform
if "win" in platform:
    parallel=0         #at pyscripter parallel is not working
else:
    parallel=1
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()














###functions###
#modolu around 0. we will right shift (+mod) it so it will be between 0 and 2*modulo, do modolu and left shift it by -mod
def modOp(num,mod):
    return (num+mod)%(2*mod)-mod




#this function just return the quantizer bins borders
#if we have left and right border and we want to put quantizer with #options
def quantizer(left,right,options):
    delta=1.0*(right-left)/options
    return np.r_[left+delta/2:right:delta]
#this function quantize a number by a giver quantizer with bins
def quantizise(mumber,quants):
    return min(quants, key=lambda x:abs(x-mumber))
###end of functions###
















#return the best error per given modulo and number of bins
#the flow goes like this:
    #random inputs
    #add dither
    #modulo
    #quantize
    #subtract the dither
    #add alpha
    #modulo
    #in -> d -> m -> q -> -d -> -> a -> m -> out
def getError(modulo,numOfBins,samples):


    #generate a quantizer with bins.
    #the quant will be the middle of the bin
    #the right and left quant will be between the modulo wall and the bin wall
    quants=quantizer(-modulo,modulo,numOfBins)
    binSize=2*(modulo)/numOfBins


    #randomizing numbers
    originalNum=np.matrix(np.random.normal(mu,sigma,samples))
    #originalNum=np.matrix(np.random.uniform(-1,1,samples))


    #adding dither:
    if ditherOn:
        dither=np.matrix(np.random.uniform(-binSize/2,binSize/2,samples))
        newNum=originalNum+dither


    #for debug, to see the dither dist
    if (False):
        debug_before_after_dither=(newNum-originalNum).A1.tolist()
        h,b=np.histogram(debug_before_after_dither,bins=10)
        b=(b[1:]+b[:-1])/2
        plot(b,h,'*-',label=modulo)
        legend(loc="best")
        show()


    #modulo:
    if moduloOn:
        newNum=modOpVec(newNum,modulo)


    #quantizer:
    #newNum=quantiziseVec(newNum,quants)
    #newNum=map(lambda x: quantizise(x,quants),newNum)
    newNum=np.matrix([min(quants, key=lambda x:abs(x-i)) for i in newNum.tolist()[0]]) #can do also by modulo by the bin size and subtract it from the number


    #subtract the dither:
    if ditherOn:
        newNum-=dither
    #modulo:
    if moduloOn:
        newNum=modOpVec(newNum,modulo)
    if ditherOn:
        if (mseInsteadOfSigma):
            error=mseVec(newNum,originalNum).tolist()[0]
            error=np.mean(error) #this is the mse
            #error=np.sqrt(error) #TODO
            alpha=1/(1+error)
        else:
            alpha=1/(1+binSize**2/12)#TODO
            #alpha=1/(1+binSize/np.sqrt(12))
        newNum*=alpha
    #calculate the error:
    error=mseVec(newNum,originalNum).tolist()[0]
    error=np.mean(error) #this is the mse


    if searchAllRange:
        return [error,modulo,numOfBins,alpha,binSize]
    else: #not maintained...
        return error






def plotErrors(errors,colorAndType,label):
    #we just plotting the mse vs number of bins
    #for now, [:,0] is to plot mse, [:,3] for bin size and [:,2] for alpha
    plot(errors.keys(),np.matrix(errors.values())[:,0],colorAndType,label=label) #TODO
    legend(loc="best")
    grid()




#this is the main program which checks all modulo sizes and all number of bins
def getErrorAllBins():
    if searchAllRange:
        #we want to find results of those values, then we will look for the minimum
        numOfBins=np.r_[2:40:1]
        modulo=np.r_[sigma/2:sigma*4:.1]
        #modulo=np.r_[0.5:4:.1]
        allOptions=np.matrix(list(itertools.product(numOfBins,modulo))).T


        #find errors
        if parallel:
            errors=np.matrix((Parallel(n_jobs=num_cores)(delayed(getErrorVec)(i[1],i[0],samples) for i in allOptions.T.tolist())))
        else:
            errors=getErrorVec(allOptions[1],allOptions[0],samples).A1


    else:#not maintain anymore...
        print fmin(getError,0.1,args=(4,samples)) #when using brent should try only >0 because at 0 you will get divide by 0


    #find the minimum mse for each number of bins: first we sort, then we take the smallest
    #we double sort by mse then by #bins
    #errors=(sorted(sorted(errors.tolist(),key=lambda e:e[0], reverse=True),key=lambda e:e[2], reverse=True))
    errors=sorted(errors.tolist(),key=lambda e:e[0], reverse=True)
    errors=(sorted(errors,key=lambda e:e[2], reverse=True))
    #take the last value for each # of bins
    errors={i[2]:[i[0],i[1],i[3],i[4]] for i in errors}
    return errors


#to see the quantizer on the gaussian.
#this function will save the plot to file
def plot_gaussian_and_quants(mu,sigma,quantizer):
    x=np.arange(mu-4*sigma,mu+4*sigma,sigma/1000)
    plot(x,norm.pdf(x,mu,sigma))
    plot(quantizer,np.zeros(len(quantizer)),"D")
    grid()
    savefig("./quantizer_on_bell_plots/"+str(len(quantizer))+"_bins.png")
    close()




















##vectorizing the functions
modOpVec=np.vectorize(modOp)
quantiziseVec=np.vectorize(quantizise)
mseVec=np.vectorize(lambda x,y:(x-y)**2)
getErrorVec=np.vectorize(getError, otypes=[list])
##end of vectorizing  the functions














#basic option
if (False):
    errors=getErrorAllBins()
    plotErrors(errors,'r',"with modulo")
    show()


#print curve per options - with or without modulo and dither
if (False):
    errors=getErrorAllBins()
    plotErrors(errors,'r',"with modulo")
    moduloOn=0
    errors=getErrorAllBins()
    plotErrors(errors,'b',"without modulo")
    show()




#for printing to files the gaussian and the quantizers
if (False):
    errors=getErrorAllBins()
    print "errors with modulo {number of bins: [error,modulo size],...}=\n",errors
    for k,v in errors.iteritems():
        plot_gaussian_and_quants(mu,sigma,quantizer(-v[1],v[1],k))


#for plotting the delta between errors with and without modulo
if (False):
    moduloOn=1
    errors=getErrorAllBins()
    with_mod=errors


    moduloOn=0
    errors=getErrorAllBins()


    delta={i:[with_mod[i][0]-errors[i][0],with_mod[i][1]] for i in with_mod.keys()}
    print "delta =",delta
    plotErrors(errors,'b',"w/o modulo")
    plotErrors(with_mod,'r',"w modulo")
    plotErrors(delta,'g',"delta")
    show()

#for plotting the div between alpha of mse or dither sigma
if (True):
    mseInsteadOfSigma=1
    errors=getErrorAllBins()
    by_mse=errors
    a=np.matrix(by_mse.values())[:,0]

    mseInsteadOfSigma=0
    errors=getErrorAllBins()
    b=np.matrix(errors.values())[:,0]
    c=b/a
    plot(c)
    show()

#for plotting the delta between alpha of mse or dither sigma
if (False):
    mseInsteadOfSigma=1
    errors=getErrorAllBins()
    by_mse=errors

    mseInsteadOfSigma=0
    errors=getErrorAllBins()


    #delta={i:[by_mse[i][0]-errors[i][0],by_mse[i][1]] for i in by_mse.keys()}
    #print "delta =",delta
    plotErrors(errors,'b',"dither sigma")
    plotErrors(by_mse,'r',"mse")
    #plotErrors(delta,'g',"delta")
    show()
