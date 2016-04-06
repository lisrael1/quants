%get bit value by given borders
function bit_value=get_bits_value(x1,x2,mu,sigma)
	if (x1 == -inf) %left tail
		bit_value=cdf('Normal',-x2,mu,sigma);
		bit_value=bit_value+(1-bit_value)/2;
		bit_value=-norminv(bit_value,mu,sigma);
	elseif (x2 == inf) %right tail
		bit_value=cdf('Normal',x1,mu,sigma);
		bit_value=bit_value+(1-bit_value)/2;
		bit_value=norminv(bit_value,mu,sigma);
	else
		bit_value=(x2+x1)/2;
	end
end
