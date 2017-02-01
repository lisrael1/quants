%literally
function [bars_values,bits_values]=get_bars_and_bits(number_of_quantoms,delta_between_quantoms,mu,sigma)%number_of_quantoms should be odd
	bars_values(1)=-inf;
	bars_values(2)=-((number_of_quantoms-1)/2)*delta_between_quantoms-delta_between_quantoms/2;
	for i= [3:number_of_quantoms+2]
		bars_values(i)=bars_values(i-1)+delta_between_quantoms;
	end		
	bars_values(number_of_quantoms+3)=inf;
	
	for i=[1:number_of_quantoms+2]
		bits_values(i)=get_bits_value(bars_values(i),bars_values(i+1),mu,sigma);
	end
end
