addpath('/a/fr-05/vol/home/grad/lisrael1/Desktop/quants/matlab')
addpath('/a/fr-05/vol/home/grad/lisrael1/Desktop/quants/matlab/funcs')
clc
number_of_quantoms=11;%you will get 2 mores at the tails. enter to odd number 
mu=0;%mu should be 0 
sigma=1;
modulo=2;

delta_between_quantoms=fminunc (@(delta_between_quantoms) analytically_check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,modulo),0);

[bars_values,bits_values]=get_bars_and_bits(number_of_quantoms,delta_between_quantoms,mu,sigma);
quantization_error=analytically_check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,modulo);

delta_between_quantoms
quantization_error
bars_values
bits_values
%modulo=0;%just that it will not plot it..
plot_quants (mu,sigma,number_of_quantoms,bars_values,bits_values,modulo);

