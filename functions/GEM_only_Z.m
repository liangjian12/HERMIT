function [rho,Z,pi,fun] = GEM_only_Z(X,Y,k,tau,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

rho = rand(n,k);
[~,idx_max] = max(rho,[],2);
rho = full(sparse([1:n]',idx_max,1));
rho(rho==0) = 0.1;
rho(rho==1) = 0.9;
rho = bsxfun(@rdivide,rho,sum(rho,2));

pi = sum(rho,1);
pi = pi/sum(pi);

opts.rho = rho;

% the default setting of regularization
if isempty(prox_param)  
     prox_param.Z.method = 'L1';
end
opts.prox_param = prox_param;
 
W = opts.fix.W;
Z = zeros(n,m,k);
Phi = opts.fix.Phi;
 

param.W = W;
param.Z = Z;
param.Phi = Phi;
param.pi = pi;
param.rho = rho;




for iter_out = 1: opts.maxIter_out
    
   param_pre = param;
    
    %M-step: parameter 
    for r = 1:k    
        opts_tmp = opts;  
        opts_tmp.rho = param.rho(:,r);
        opts_tmp.Phi = param.Phi(:,:,r);
        prox_param.Z.param = tau * param.pi(r)^(gamma) ;
        opts_tmp.prox_param = prox_param;
        opts_tmp.fix.W = param.W(:,:,r);
 
        if prox_param.Z.fix_feature_flag
            if mod(iter_out,10) == 1
                opts_tmp.prox_param.Z.fix_feature_flag = false;
                opts_tmp.prox_param.Z.fix_feature = [];
            else
                opts_tmp.prox_param.Z.fix_feature_flag = true;
                opts_tmp.prox_param.Z.fix_feature = abs(param.Z(:,:,r))>eps;
            end
        else
            opts_tmp.prox_param.Z.fix_feature_flag = false;
            opts_tmp.prox_param.Z.fix_feature = [];
        end
        
        [param.Z(:,:,r)] = apg_W_GD_only_Z(X,Y,opts_tmp);
    end

    %E-step
    [fun_c(iter_out),rho] = estimate_fun_only_Z(X,Y,param.W,param.Z,param.Phi,param.pi,Omega,opts);
    
    param.rho = rho;
 
     %M-step: pi
    pi = sum(rho,1);
    pi = pi/sum(pi);
   
    param.pi = pi;

    if iter_out>1
        d1 = param_dist_without_Z(param_pre,param,'max');
        d2 = abs(fun_c(iter_out)-fun_c(iter_out-1))/(1+abs(fun_c(iter_out)));
        if d2 <= opts.stop_param.fun && d1 <= opts.stop_param.param
            break
        end
    end
     
end

 
Z = param.Z;

pi = param.pi;
 
rho = param.rho;
 
end