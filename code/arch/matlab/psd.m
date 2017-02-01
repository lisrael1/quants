sigma=13;
mu=0;
modulo=25;
xrange = [(-3*sigma):.1:(3*sigma)];
for modulo=[10:40]
	figure
	xrange = [(-modulo):.1:(modulo)];
	%a= (normpdf(xrange,mu,sigma)+normpdf(xrange,mu-modulo,sigma)+normpdf(xrange,mu+modulo,sigma)+normpdf(xrange,mu-2*modulo,sigma)+normpdf(xrange,mu+2*modulo,sigma))/5;
	a=normpdf(xrange,mu,sigma);
	b=a;
	plot (xrange,a);hold on;
	title(modulo)
	for i=[1:15]
		a=normpdf(xrange,mu-2*i*modulo,sigma)+normpdf(xrange,mu+2*i*modulo,sigma);
		b=b+a;
		%plot (xrange,a,xrange,normpdf(xrange,mu,sigma),xrange,normpdf(xrange,mu-modulo,sigma),xrange,normpdf(xrange,mu+modulo,sigma))
		plot (xrange,a);hold on;
	end
	%figure
	plot (xrange,b,'.')
	title(modulo)
end

