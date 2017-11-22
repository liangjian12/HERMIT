function mu = link_fun_mu(g,options)

 

if strcmp(options,'gauss')
    
   
%     b =  theta.^2/2;
    
    mu = g;
   
    
elseif strcmp(options,'bnl')
    
   
%     b =  theta + log(1+exp(-theta));
    mu =  1./(1+exp(-g));
  
    
elseif strcmp(options,'poiss')    
    
    
%     b = exp(theta);
%    mu = exp(g);
     mu = exp(g);
    
end






end