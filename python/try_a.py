
execfile("functions/functions.py")
q=quantizer(number_of_quants=21,bin_size=1)

A=m([[1,-1,1,-1],[1,2,-1,-2],[-1,-1,1,1],[1,3,7,-10]]).T
A=m([[1,1,1,1],[1,2,-1,2],[-1,-1,2,4],[1,3,1,-1]]).T
print linalg.det(A)
a=OrderedDict()
mod_size=10
a['orign']=m([101,-102,-99,100])
a['aXA']=a['orign']*A
a['quant']=q.quantizise(a['orign'])
a['modulo']=mod_op(a['quant'],mod_size)
a['after a']=a['modulo']*A
a['mod2']=mod_op(a['after a'],mod_size)
a['invrs']=a['mod2']*A.I
print "\n".join([str(k)+"\t:"+str(v) for k,v in a.iteritems()])



exit()

##print A*a.T
c=m([[ -2],[406],[ -4],[402]])
d=m([[ -2],[6],[ -4],[2]])
print A.T*d
