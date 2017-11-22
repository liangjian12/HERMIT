function [rho,pi,W,Phi,fun_c] = GEM_without_Z_multi_comp(X,Y,k,lambda,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

% the default setting of regularization
if isempty(prox_param)  
    prox_param.W.method = 'L1';  % regularization method, options are : 'L2' for L2 norm; 'L1' for L1 norm;  'EN' for Elastic Net;  'GS' for Group Sparsity, which is only valid for matrix;     
end
opts.prox_param = prox_param;

if isempty(opts.sep_pen.lambda_set)
    opts.sep_pen.lambda_set = opts.lambda * ones(1,3);
end


rho = rand(n,k);
[~,idx_max] = max(rho,[],2);
rho = full(sparse([1:n]',idx_max,1));
rho(rho==0) = 0.1;
rho(rho==1) = 0.9;
rho = bsxfun(@rdivide,rho,sum(rho,2));

pi = sum(rho,1);
pi = pi/sum(pi);

if opts.warm_start.flag
    
W = opts.init.W;
Phi = opts.init.Phi;
 
 
else
 
W = randn(d,m,k) * opts.initial_scale;  
 
Phi = zeros(m,m,k);
for r = 1:k
    Phi(:,:,r) = eye(m);
end

end

param.W = W;
param.Phi = Phi;
param.pi = pi;
param.rho = rho;
opts.rho = rho; 

opts.stop_param.fun  = 2.5e-5*m*k;
opts.stop_param.param = 2.5e-2*m*k;

 
for iter_out = 1: opts.maxIter_out
    
   opts.iter_out_gem = iter_out;
    
   param_pre = param;
    
    %M-step: parameter 
      
    opts.rho = param.rho;
    opts.Phi = param.Phi;
    prox_param.W.param = lambda .* param.pi.^(gamma) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
 
    if opts.sep_pen.flag
        prox_param.W.sep.param_set =    opts.sep_pen.lambda_set(:) * param.pi.^(gamma) ; %3,k 
    end

    if prox_param.W.fix_feature_flag
        if mod(iter_out,10) == 1
            prox_param.W.fix_feature_flag = false;
            prox_param.W.fix_feature = zeros(d,m,k);
        else
            prox_param.W.fix_feature_flag = true;
            prox_param.W.fix_feature = abs(param.W_cube)>eps;
        end
    else
        prox_param.W.fix_feature_flag = false;
        prox_param.W.fix_feature = zeros(d,m,k);
    end

    opts.prox_param = prox_param;
    opts.init.W = W;
   
    if opts.sep_pen.flag
        [param.W,~] = apg_W_GD_without_Z_multi_comp_sep(X,Y,opts);
    else
        [param.W,~] = apg_W_GD_without_Z_multi_comp(X,Y,opts);
    end

    %E-step
    [fun_c(iter_out),rho] = estimate_fun_without_Z(X,Y,param.W,param.Phi,param.pi,Omega,opts);
    
    param.rho = rho;
 
 
    %M-step: pi
    pi = sum(rho,1);
    pi = pi/sum(pi);
    
    
    if gamma>0
        eta = 0.1;        
        for r = 1:k
            tmp = abs(param.W(:,:,r));
            L1_W(r) = sum(tmp(:));
        end        
        L_pre = loss_pi_without_Z(rho,param_pre.pi,gamma,lambda,L1_W);
        L = L_pre;
        
        cont = 1;
        while L>=L_pre && cont < 10;        
            pi_new = param_pre.pi + eta^cont * (pi - param_pre.pi);
            L = loss_pi_without_Z(rho,pi_new,gamma,lambda,L1_W);
            cont = cont + 1;
        end
        
        pi = pi_new;
    end        
    
    param.pi = pi;
    
     if iter_out>1
        d1 = param_dist_without_Z(param_pre,param,'max');
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
Phi = param.Phi;
pi = param.pi;
rho = param.rho;
 
end