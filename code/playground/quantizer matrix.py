import numpy as np
'matrix quantizer'

dim=2
x=np.mat(np.random.normal(0,0.5,[5,dim])).round(4)#round is just for each reading, but you dont have to use it

quant_size=[0.1,0.01]#or [0.2] for same quantization for all dims
offset=[0.05,0.] #if you want quantizer not to start at 0, like -0.1,-0.05,0.05,0.1. you can also put [0.1] for same value for all dims

r=np.mat(np.identity(dim)*quant_size)

print "x:\n",x
print "x:\n",x+offset
print "quantizedx:\n",((x+offset)*r.I).round()*r-offset