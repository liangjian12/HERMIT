function [rho,Z,fun] = opti_only_Z(X,Y,k,tau,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

rho = exprnd(1,n,k);
rho = bsxfun(@rdivide,rho,max(rho,[],2));
rho(rho~=1) = 0.1;
rho(rho==1) = 0.9;
rho = bsxfun(@rdivide,rho,sum(rho,2));

pi = opts.fix.pi;

opts.rho = rho;

% the default setting of regularization
if isempty(prox_param)  
     prox_param.Z.method = 'L1';
end
opts.prox_param = prox_param;
 
W = opts.fix.W;
Z = {};
Phi = opts.fix.Phi;
for r = 1:k
    Z{r} = zeros(n,m);
end

param.W = W;
param.Z = Z;
param.Phi = Phi;
param.pi = pi;
param.rho = rho;
 


   param_pre = param;
    
    %M-step: parameter 
    for r = 1:k    
        opts_tmp = opts;  
        opts_tmp.rho = param.rho(:,r);
        opts_tmp.Phi = param.Phi{r};
        prox_param.Z.param = tau * param.pi(r)^(gamma) ;
        opts_tmp.prox_param = prox_param;
        if  prox_param.Z.ada_lasso_flag
            opts_tmp.prox_param.Z.ada_lasso_weight = prox_param.Z.ada_lasso_weight{r};
        end
        [param.Z{r}] = apg_W_GD_only_Z(X,Y,opts_tmp);
    end
 
 
    
    %E-step
    [fun,fun_c,rho] = estimate_fun_with_Z(X,Y,param.W,param.Z,param.Phi,param.pi,Omega,opts);
    
    param.rho = rho;
 
    
     
     disp([iter_out,fun/1000,fun_c/1000])     
 
 
Z = param.Z;
 
 
rho = param.rho;
 
end