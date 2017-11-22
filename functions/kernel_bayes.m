function K = kernel_bayes(D,method)

if strcmp(method,'gauss')
    
    K = exp(-D.^2/2);
    
elseif strcmp(method,'ard')
    D2 = D.^2;
    K = (1 + sqrt(5*D2)+ 5 * D2/3 ).*exp( - sqrt(5) * D);
    
end



end