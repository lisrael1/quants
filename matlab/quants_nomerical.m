function quants ()
    clc
    global debug;
    debug=0;
    number_of_quantoms=9;
    %delta_between_quantoms=0.01; %how many bits you have
    mu=0;
    sigma=4;
    how_many_x=500;
    
    %delta_between_quantoms = fminsearch(@(delta_between_quantoms) check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,how_many_x),0.01)
    %delta_between_quantoms = fminunc(@(delta_between_quantoms) check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,how_many_x),0.01)
    x=[];
    e=[];
    for delta_between_quantoms = [0:sigma/50:sigma]
    %for i =[0:5]
        error=check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,how_many_x);
        %fprintf ('for delta %f we get error %f\n',delta_between_quantoms,error) 
        %fprintf ('%f\t%f\n',delta_between_quantoms,error) 
        x=[x delta_between_quantoms];
        e=[e error];
    end
    p=polyfit(x,e,2);
    y=polyval(p,x);
    delta_between_quantoms = fminunc (@(x) polyval(p,x),0)
    figure;
    plot(x,y,x,e) 
    
    plot_quants (mu,sigma,number_of_quantoms,delta_between_quantoms)
end

function dsp(s)
    global debug;
    if debug
        disp(s)
    end
end
function prnt(s,x)
    global debug;
    if debug
        fprintf(s,x)
    end
end

function error = check_error(number_of_quantoms,delta_between_quantoms,mu,sigma,how_many_x)
    error=0;
    tail_value=get_tail_value(number_of_quantoms,delta_between_quantoms,mu,sigma);
    x=normrnd(mu,sigma,[1 how_many_x]);
    for n = x
        dsp('===================================')
        quantized=quantization(n,number_of_quantoms,delta_between_quantoms,tail_value);
        current_error=power((n-quantized),2);
        %current_error=abs(n-quantized);
        prnt('current error is = %f\n',current_error)
        error=error+current_error/how_many_x;
    end
    dsp('===================================')
    prnt('total error is = %f\n',error)
end

function quantized = quantization(x,number_of_quantoms,delta_between_quantoms,tail_value)
    seperators_one_side=floor(number_of_quantoms/2);
    prnt('calculating output for %f\n',x)
    if (abs(x)<(seperators_one_side*delta_between_quantoms))
        dsp('x is inside our range')
        quantized=sign(x)*floor(abs(x/delta_between_quantoms))*delta_between_quantoms+delta_between_quantoms/2;
    else
        if(x>0)
            dsp('y outside our range')
            quantized=tail_value;
        else
            dsp('y outside our range')
            quantized=-tail_value;
        end
    end
    prnt('y = %f\n',quantized)
end

function tail_val = get_tail_value (number_of_quantoms,delta_between_quantoms,mu,sigma)
    last_quant_value=floor(number_of_quantoms/2)*delta_between_quantoms;
    last_quant_probability=cdf('Normal',last_quant_value,mu,sigma);
    tail_val=norminv(((1-last_quant_probability)/2)+last_quant_probability,mu,sigma);
    %sanity check: y=-1*norminv(last_quant_probability,mu,sigma)
    prnt('last quant value = %f\n',last_quant_value);
    prnt('last quant probability = %f\n',last_quant_probability);
    prnt('so the tail value will be = %f\n',tail_val);
end

function plot_quants (mu,sigma,number_of_quantoms,delta_between_quantoms)
    xrange = [(-3*sigma):.1:(3*sigma)];
    norm = normpdf(xrange,mu,sigma);
    y=[];
    bar=[];
    for i = [0:floor(number_of_quantoms/2)]
        bar=[bar i*delta_between_quantoms -i*delta_between_quantoms];
        y=[y 0 0];
    end
    figure
    plot(xrange,norm,bar,y,'^')        
end
