#!/usr/bin/python
execfile("functions/functions.py")

q=quantizer(number_of_quants=1000,bin_size=0.02)
##print q.quantizise([4,3])

Hz=800 #sample Hz
x = np.linspace(0, 1, Hz)[:-1]#-1 for complete sine with no patial cycle
q_in=np.sin(50.0 * 2.0*np.pi*x)
q_out=q.quantizise(q_in).A1

data=pd.DataFrame(zip(x,q_in,q_out),columns=["x","q_in","q_out"])
data['mse']=data.apply(lambda row: (row.q_out-row.q_in)**2,axis=1)



freq = np.abs(np.fft.fft(data.q_out))
N=len(freq)
freq=2.0/N*freq[0:N/2]
xf = np.linspace(0.0, Hz/2.0, N/2)

fft=pd.DataFrame(zip(xf,freq),columns=["xf","amp"])

current=max(fft.amp)
al=sum(fft.amp)
print "sine power =",current
print "total power including sine freq =",al
print "total power not including sine freq =",al-current
print "snr = ",current/(al-current)
print "current/all =",current/(al),"%"
print "sqrt mse =",sqrt(sum(data.mse))



exit()
subplot(212)
plot(fft.xf,fft.amp,'-')
subplot(211)
plot(x,data.q_in)
plot(x,data.q_out)
show()