addpath('/a/fr-05/vol/home/grad/lisrael1/Desktop/quants/matlab')
addpath('/a/fr-05/vol/home/grad/lisrael1/Desktop/quants/matlab/funcs')
clc
warning('off','all')
number_of_quantoms=3;%you will get 2 mores at the tails. enter to odd number 
mu=0;%mu should be 0 
sigma=1;
%mod=[0.5:0.1:3];
modulo=100;
i=1;
qua=[3:123];
err=qua;
options = optimoptions(@fminunc,'Display','none');
for  number_of_quantoms=qua
	delta_between_quantoms=fminunc (@(delta_between_quantoms) analytically_check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,modulo),0,options);

	[bars_values,bits_values]=get_bars_and_bits(number_of_quantoms,delta_between_quantoms,mu,sigma);
	quantization_error=analytically_check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,modulo);

	delta_between_quantoms;
	quantization_error;
	modulo;
	err(i)=quantization_error;
	i=i+1
	%plot_quants (mu,sigma,number_of_quantoms,bars_values,bits_values,modulo);
end
plot(qua,err,'.')
%bars_values
%bits_values
%plot_quants (mu,sigma,number_of_quantoms,bars_values,bits_values,modulo);

