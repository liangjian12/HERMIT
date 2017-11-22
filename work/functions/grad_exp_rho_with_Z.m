function [dW,dZ] = grad_exp_rho_with_Z(X,Y,W,Z,Phi,Omega,rho,options)
[n,d] = size(X);
[n,m] = size(Y);

dZ_save = zeros(size(Z));

idx = rho > 0;
n = sum(rho);
X = X(idx,:);
Y = Y(idx,:);
Z = Z(idx,:);
Omega = Omega(idx,:);
rho = rho(idx);

G = X*W +Z;


if strcmp(options,'gauss')
    
    mu = link_fun_mu(G,options);  
    err = bsxfun(@times,Y*Phi-mu,rho);
    dZ = err.*Omega;
    dW = X'*dZ;
 
     
elseif strcmp(options,'bnl')
    
    mu = link_fun_mu(G,options);  
 
 
    err = Y  - mu;
    err = bsxfun(@times,err,rho);
    
    idx_pos = double(Y==1 & Omega == 1);
    idx_neg = double(Y==0 & Omega == 1);
    n_pos = sum(bsxfun(@times,idx_pos,rho),1);
    n_neg = sum(bsxfun(@times,idx_neg,rho),1);
    w_pos = n./(2*n_pos);
    w_neg = n./(2*n_neg);
    
    err_pos = bsxfun(@times,err,w_pos) .* idx_pos;
    err_neg = bsxfun(@times,err,w_neg) .* idx_neg;
    
    err = err_pos+err_neg;
 
    dZ = err.*Omega;
    dW = X'*dZ;
 
    
elseif strcmp(options,'poiss')    
    
    mu = link_fun_mu(G,options); 
    err = bsxfun(@times,Y-mu,rho);
    dZ = err.*Omega;
    dW = X'*dZ;
 
    
end

% n_omega = max(1,sum(Omega,1));
% dZ = bsxfun(@rdivide,dZ,n_omega);
% dW = bsxfun(@rdivide,dW,n_omega);

dZ_save(idx,:) = dZ;
dZ = dZ_save;
 
dZ = dZ/size(X,1);
dW = dW/size(X,1);


end