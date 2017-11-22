function [dW,dZ] = grad_exp_rho(X,Y,W,Z,Phi,Omega,rho,options)
[n,d] = size(X);
[n,m] = size(Y);


G = X*W + Z;


if strcmp(options,'gauss')
    
    mu = link_fun_mu(G,options);  
    err = bsxfun(@times,Y-mu,rho);
    dZ = (err/Phi).*Omega;
    dW = X'*dZ;
 
     
elseif strcmp(options,'bnl')
    
    mu = link_fun_mu(G,options);  
    err = bsxfun(@times,Y-mu,rho);
    dZ = err.*Omega;
    dW = X'*dZ;
 
    
elseif strcmp(options,'poiss')    
    
    mu = link_fun_mu(G,options); 
    err = bsxfun(@times,Y-mu,rho);
    dZ = err.*Omega;
    dW = X'*dZ;
 
    
end


end