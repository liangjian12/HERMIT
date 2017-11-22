function [dW] = grad_exp_rho_without_Z(X,Y,W,Phi,Omega,rho,options)
[n,d] = size(X);
[n,m] = size(Y);

idx = rho > 0;
n = sum(rho);
X = X(idx,:);
Y = Y(idx,:);
Omega = Omega(idx,:);
rho = rho(idx);

G = X*W ;
 

if strcmp(options,'gauss')
    
    mu = link_fun_mu(G,options); 

    err = Y*Phi - mu;
    err = bsxfun(@times,err,rho);
    tmp = err.*Omega;
    dW = X'*tmp;
 
     
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
 
    tmp = err.*Omega;
    dW = X'*tmp;
 
    
elseif strcmp(options,'poiss')    
    
    mu = link_fun_mu(G,options); 
 
    err = Y  - mu;
    err = bsxfun(@times,err,rho);
    tmp = err.*Omega;
   
    dW = X'*tmp;
 
    
end

% n_omega = max(1,sum(Omega,1));
% dW = bsxfun(@rdivide,dW,n_omega);

% dW = dW/n;

dW = dW/size(X,1);
end