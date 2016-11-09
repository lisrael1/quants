'''
for now, the quantizer is around 0
example code:
	q=quantizer(0.1,4)
	q=quantizer(0.1,4,10)
	q=quantizer(all_quants=[-0.15,-0.05,0.05,0.15])
	print q
'''
class quantizer():
	def __init__(self,bin_size=None,number_of_quants=None,max_val=None,all_quants=None,mu=0,sigma=1,modulo_edge_to_edge=None):
		self.mu=mu
		self.sigma=float(sigma)
		self.var=self.sigma**2
		if bin_size!=None and number_of_quants!=None and max_val==None:
			self.bin_size=float(bin_size)
			self.number_of_quants=number_of_quants
			self.max_val=self.bin_size*(self.number_of_quants-1)/2.0
			self.all_quants=[i*self.bin_size-self.max_val for i in range(number_of_quants)]
			self.modulo_edge_to_edge=self.max_val*2+self.bin_size
		elif number_of_quants!=None and modulo_edge_to_edge!=None:
			self.number_of_quants=int(number_of_quants)
			self.modulo_edge_to_edge=float(modulo_edge_to_edge)
			self.bin_size=self.modulo_edge_to_edge/float(self.number_of_quants+1)
			self.max_val=self.bin_size*(self.number_of_quants-1)/2.0
			self.all_quants=[i*self.bin_size-self.max_val for i in range(number_of_quants)]
		elif all_quants!=None:
			self.all_quants=all_quants
			self.bin_size=all_quants[1]-all_quants[0]
			self.number_of_quants=len(all_quants)
			self.max_val=self.all_quants[-1]
			self.modulo_edge_to_edge=self.max_val*2+self.bin_size
	"""
	this function quantize a number by a given quantizer with bins
	"""
	def quantizise(self,numbers):
		if (self.number_of_quants>1000):
			return numbers
		#rounding to the quantizer:
		q=rint((numbers+self.max_val)/(1.0*self.bin_size))*self.bin_size-self.max_val
		#taking the edges:
		q=mat([self.max_val if i>self.max_val else -self.max_val if i<-self.max_val else i for i in q.A1]).reshape(q.shape)
		return q
	def plot_pdf_quants(self):
		x=arange(-self.max_val-2*self.bin_size,self.max_val+2*self.bin_size,self.sigma/1000.0)
		plot(x,norm.pdf(x,self.mu,self.sigma),label="pdf")
		plot(self.all_quants,zeros(self.number_of_quants),"D",label="self")
		legend(loc="best")
		error=analytical_error(self.bin_size,self.number_of_quants,self.mu,self.sigma)
		title("#bins="+str(self.number_of_quants)+", bin size="+str(self.bin_size)+"\nself.mu="+str(self.mu)+", self.sigma="+str(self.sigma)+", error="+str(error))
		grid()
		show()
	def __str__(self):
		print "-----"
		print "bin_size:",self.bin_size
		print "number_of_quants:",self.number_of_quants
		print "max_val:",self.max_val
		print "modulo_edge_to_edge:",self.modulo_edge_to_edge
		print "all_quants:",self.all_quants
		return ""


'''
get quants, mu and sigma and return the analytic error for those values
note that mu should be around 0 because of the mod in the mse_for_single_dot
examples:
	print analytical_error(2.5,5,0,1)
	plot([analytical_error(i/10.0,5,0,1) for i in range(1,100)])
	show()
'''
def analytical_error(bin_size,number_of_quants,mu,sigma):
	q=quantizer(bin_size=bin_size,number_of_quants=number_of_quants)
	def quantizise_single(x,quantizer):
		#taking max and min values to be at the last bins
		#rounding to the quantizer:
		q=rint((x+quantizer.max_val)/(1.0*quantizer.bin_size))*quantizer.bin_size-quantizer.max_val
		#cutting the edges to the mas quantizer value:
		if q>quantizer.max_val:
			q=quantizer.max_val 
		if q<-quantizer.max_val:
			q=-quantizer.max_val 
		return q
	def mse_for_single_dot(x,mu,sigma,quantizer):
		simple_one=0
		if simple_one:
			return norm(mu,sigma).pdf(x)*((x-quantizise_single(x,quantizer))**2)
		else:
			#TODO: add dither?
			#dither=random.uniform(0,quantizer.bin_size)
			return norm(mu,sigma).pdf(x)*((x-quantizise_single(mod_op(x,quantizer.modulo_edge_to_edge),quantizer))**2)
	error=quad(mse_for_single_dot,-10*sigma,10*sigma,args=(mu,sigma,q))[0]
	see_itterations=0
	if see_itterations:
		print q.bin_size,error
	return error

'''
look for best quantizer by number of quants and normal dist args
example:
	number_of_quants=5
	sigma=1.0
	q=find_best_quantizer(number_of_quants,sigma)
	print q
	q.plot_pdf_quants()
'''
def find_best_quantizer(number_of_quants,sigma,mu=0):
	#we will start looking from sigma 
	start_looking_from=4*sigma/number_of_quants
	stop_looking_at_bin_size_error=sigma/(100.0*number_of_quants)
	bin_size=fmin(analytical_error,start_looking_from,xtol=stop_looking_at_bin_size_error,ftol=sigma*100,args=(number_of_quants,mu,sigma),disp=False).tolist()[0]
	return quantizer(bin_size=bin_size,number_of_quants=number_of_quants,mu=mu,sigma=sigma)
	#print brent(analytical_error,args=(1000,0,1))
	#print minimize(analytical_error,(1,0.1),method='TNC',args=(1000,0,1))
	
