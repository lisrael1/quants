from multiprocessing import Pool
from time import sleep
def fo((a,b)):#the argument has to be in another () because the map passing the sub list as 1 argument
	print "a",a,"b",b
	return ""
def foo(a):
	print "a",a
	return ""
class cl():
	def __init__(self,a,b):
		self.a,self.b=a,b
	def hi(self):
		print "hi"
		return ""
def fooo(c):
	print c.a,c.b
	return ""
def n(a):#helf function to next steps...
	a.hi()
	return a
p = Pool()
	#this line has to be after the function declaration!
p.map(fo,[[1,2],[4,5]]
	)#you cannot use dictionary here, it will just take the keys instead of reading the dictionary
p.map(foo,[1,24,4])
p.map(fooo,[cl(99,33)])
p.map(None,[cl(99,33).hi()])
	#you can run with no function but it's a bad idea because it will first prepare the list and then send it to the empty function, but preparing the list is serial so it will run hi at serial. this is how you should use this:
c=p.map(n,[cl(99,33)])
	#you cannot enter lambda as a function... 
	#if you want your data back you have to use the return value because map is only coping the object, not by reference


#progress:
#first method:
for i,j in enumerate(p.imap_unordered(abs,[.1,.2,.1])):
	print i,j
		#this way we can count how many results we currently have

#second method (not working good, for some reason it stops the parallel work):
d=p.imap_unordered(abs,[.1,.2,2,5])
old=0
while d._index<d._length or not d._index:
		#d._length return how many items there are...
		#for some reasone before the first result d._length is None so i added the or here
		#note that it's abs function, very quick so we will not see the progression here..
	while old==d._index:
		sleep(0.5)
	old=d._index
	print "now we have", d._index,"results"
print "results: ",list(d)
	#you have to cast it from pool item to list 


