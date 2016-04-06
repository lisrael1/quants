%get quantization squared error between given borders
function err=analytical_error(x1,x2,mu,sigma,modulo)%x2 always bigger than x1
	bit_value=get_bits_value(x1,x2,mu,sigma);
	%we need to do integral on f(x)*(x-quant_value) so for each x we will have its error. the error is at power 2
	fun=@(x) (exp(-0.5*((x-mu)/sigma).^2)/sqrt(2*3.1415)).*(sign(x).*mod(abs(x),modulo)-bit_value).^2;%if you put modulo=sigma*1000 it's like not using modulo...
	err=integral(fun,x1,x2);
end
