%plot results
function plot_quants (mu,sigma,number_of_quantoms,bars_values,bits_values,modulo)
    xrange = [(-3*sigma):.1:(3*sigma)];
    norm = normpdf(xrange,mu,sigma);
    figure
    hold on
    %plot(xrange,norm,bars_values,zeros(1,number_of_quantoms+3),'+',bits_values,zeros(1,number_of_quantoms+2),'^',[-modulo,modulo],[0,0],'*')
    plot(xrange,norm,bars_values,zeros(1,number_of_quantoms+3),'+');
    plot(bits_values,zeros(1,number_of_quantoms+2),'^');
    plot([-modulo,modulo],[0,0],'*');
    title(['mu=',num2str(mu),', sigma=',num2str(sigma),', modulo',num2str(modulo)]);
end
