function [dW,dZ] = grad_exp(X,Y,W,Z,Phi,Omega)
[n,d] = size(X);
[n,m] = size(Y);


G = X*W + Z;


if strcmp(options,'gauss')
    
    mu = link_fun_mu(G,options);  
    dZ = ((Y-mu)/Phi).*Omega;
    dW = X'*dZ/n;
    dZ = dZ/n;
     
elseif strcmp(options,'bnl')
    
    mu = link_fun_mu(G,options);  
    dZ = (Y-mu).*Omega;
    dW = X'*dZ/n;
    dZ = dZ/n;
    
elseif strcmp(options,'poiss')    
    
    mu = link_fun_mu(G,options);  
    dZ = (Y-mu).*Omega;
    dW = X'*dZ/n;
    dZ = dZ/n;
    
end


end