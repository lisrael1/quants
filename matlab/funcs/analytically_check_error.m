%inputs are the number of quantoms, delta between them and normal dist and the output is overall the quantization error
function quantization_error=analytically_check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,modulo)
	[bars_values,bits_values]=get_bars_and_bits(number_of_quantoms,delta_between_quantoms,mu,sigma);
	quantization_error=0;
	for i=[1:number_of_quantoms+2]%one less because using i+1
		quantization_error=quantization_error+analytical_error(bars_values(i),bars_values(i+1),mu,sigma,modulo);
	end
end
