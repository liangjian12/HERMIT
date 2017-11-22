function [rho,pi,W,Z,Phi,fun_c] = GEM_with_Z(X,Y,k,lambda,tau,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

rho = exprnd(1,n,k);
rho = bsxfun(@rdivide,rho,max(rho,[],2));
rho(rho~=1) = 0.1;
rho(rho==1) = 0.9;
rho = bsxfun(@rdivide,rho,sum(rho,2));

pi = sum(rho,1);
pi = pi/sum(pi);

opts.rho = rho;

% the default setting of regularization
if isempty(prox_param)  
    prox_param.W.method = 'L1';  % regularization method, options are : 'L2' for L2 norm; 'L1' for L1 norm;  'EN' for Elastic Net;  'GS' for Group Sparsity, which is only valid for matrix;     
    prox_param.Z.method = 'L1';
end
opts.prox_param = prox_param;

if isempty(opts.sep_pen.lambda_set)
    opts.sep_pen.lambda_set = opts.lambda * ones(1,3);
end
if isempty(opts.sep_pen.tau_set)
    opts.sep_pen.tau_set =  opts.tau * ones(1,3);
end
 
if opts.warm_start.flag
    
W = opts.init.W;
Phi = opts.init.Phi;
Z = zeros(n,m,k);
 
else
 
W = randn(d,m,k) * opts.initial_scale;  
Z = zeros(n,m,k);
Phi = zeros(m,m,k);
for r = 1:k
    Phi(:,:,r) = eye(m);
end

end

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
        prox_param.W.param = lambda * param.pi(r)^(gamma) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
        prox_param.Z.param = tau(r) * param.pi(r)^(gamma) ;
        
        if opts.sep_GD.flag
            prox_param.W.sep.param_set = opts.sep_GD.lambda_set  * param.pi(r)^(gamma) ;  
            prox_param.Z.sep.param_set = opts.sep_GD.tau_set  * param.pi(r)^(gamma) ;  
        end
        
        
      
        
        if prox_param.W.fix_feature_flag
            if mod(iter_out,10) == 1
                prox_param.W.fix_feature_flag = false;
                prox_param.W.fix_feature = zeros(size(param.W(:,:,r)));
            else
                prox_param.W.fix_feature_flag = true;
                prox_param.W.fix_feature = abs(param.W(:,:,r))>eps;
            end
        else
            prox_param.W.fix_feature_flag = false;
            prox_param.W.fix_feature = zeros(size(param.W(:,:,r)));
        end
        
        if prox_param.Z.fix_feature_flag
            if mod(iter_out,10) == 1%false %
                prox_param.Z.fix_feature_flag = false;
                prox_param.Z.fix_feature = zeros(size(param.Z(:,:,r)));
            else
                prox_param.Z.fix_feature_flag = true;
                prox_param.Z.fix_feature = abs(param.Z(:,:,r))>eps;%opts.train_Z_idx;%
            end
        else
            prox_param.Z.fix_feature_flag = false;
            prox_param.Z.fix_feature = zeros(size(param.Z(:,:,r)));
        end
        
        opts_tmp.prox_param = prox_param;
        
        if opts.sep_GD.flag
            [param.W(:,:,r),param.Z(:,:,r),param.Phi(:,:,r)] = apg_W_GD_with_Z_sep(X,Y,opts_tmp);
        else
            [param.W(:,:,r),param.Z(:,:,r),param.Phi(:,:,r)] = apg_W_GD_with_Z(X,Y,opts_tmp);
        end
    end
 
    %E-step
    [fun_c(iter_out),rho] = estimate_fun_with_Z(X,Y,param.W,param.Z,param.Phi,param.pi,Omega,opts);
    
    param.rho = rho;
 
 
    %M-step: pi
    pi = sum(rho,1);
    pi = pi/sum(pi);
    
    
    if gamma>0
        eta = 0.1;        
        for r = 1:k
            tmp = abs(param.W(:,:,r));
            L1_W(r) = sum(tmp(:));
            tmp = abs(param.Z(:,:,r));
            L1_Z(r) = sum(tmp(:));
        end        
        L_pre = loss_pi_with_Z(rho,param_pre.pi,gamma,lambda,L1_W,tau,L1_Z);
        L = L_pre;
        
        cont = 1;
        while L>=L_pre && cont < 10;        
            pi_new = param_pre.pi + eta^cont * (pi - param_pre.pi);
            L = loss_pi_with_Z(rho,pi_new,gamma,lambda,L1_W,tau,L1_Z);
            cont = cont + 1;
        end
        
        pi = pi_new;
    end        
    
    param.pi = pi;
    
     if iter_out>1
        d1 = param_dist_with_Z(param_pre,param,'max');
        d2 = abs(fun_c(iter_out)-fun_c(iter_out-1))/(1+abs(fun_c(iter_out)));
        if d2 <= opts.stop_param.fun && d1 <= opts.stop_param.param
            break
        end
    end
    
%     if mod(iter_out,10) == 0    
%      disp([iter_out,fun(iter_out)/1000,fun_c(iter_out)/1000])     
%     end
     
end

W = param.W;
Z = param.Z;
Phi = param.Phi;
pi = param.pi;
rho = param.rho;
 
end