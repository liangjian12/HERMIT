function [rho,pi,W,Phi,W_k1,Phi_k1,pi_outlier,fun_c] = GEM_without_Z_outlier(X,Y,k,lambda,gamma,prox_param,opts)

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

% rho = rand(n,k);
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

pi_outlier = ones(1,m)-0.05;

param.W = W;
param.Phi = Phi;
param.pi = pi;

w = permute(rho,[1,3,2]);
w = repmat(w,[1 m 1]);

param.rho = bsxfun(@times,w,pi_outlier);
param.pi_outlier = pi_outlier;

rho_global = rho;

param_k1.W = W(:,:,1);
param_k1.Phi = Phi(:,:,1);
param_k1.rho = sum(bsxfun(@times,w,1-pi_outlier),3);

opts.rho = rho;
param.alpha = ones(1,m)/m;

opts.stop_param.fun  = 2.5e-5*m*k;
opts.stop_param.param = 2.5e-2*m*k;

% opts.stop_param.fun  = 2.5e-5*k;
% opts.stop_param.param = 2.5e-2*k;
 
 
fun_c = zeros(1,opts.maxIter_out);
for iter_out = 1: opts.maxIter_out
    
    opts.iter_out_gem = iter_out;
    
   
    pi_outlier = ones(1,m)-0.001;
    
    
%     if iter_out > 1
%         opts.warm_start.flag = true;
%     end
%     
   param_pre = param;
   
    %M-step: parameter 
    for r = 1:k    
        opts_tmp = opts;  
        opts_tmp.rho = param.rho(:,:,r);
        opts_tmp.Phi = param.Phi(:,:,r); 
    
        if opts.sep_pen.flag  
            prox_param.W.sep.param_set = opts.sep_pen.lambda_set  * param.pi(r)^(gamma) ;  
        else
            prox_param.W.param(1:3) = lambda * param.pi(r)^(gamma) * ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
        end
        
        if opts.flag_decomp           
            prox_param.W.param_decomp{1}(1:3) = opts.lambda * param.pi(r)^(gamma) * ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization  
            prox_param.W.param_decomp{2}(1:3) = opts.lambda2 * param.pi(r)^(gamma) * ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization               
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
        
        opts_tmp.prox_param = prox_param;
        opts_tmp.init.W = W(:,:,r);
        
        
        if opts.sep_pen.flag
            [param.W(:,:,r),~] = apg_W_GD_without_Z_sep(X,Y,opts_tmp);
        else
            [param.W(:,:,r),~] = apg_W_GD_without_Z(X,Y,opts_tmp); 
        end
       
    end
    
    
 
%     param_k1.rho = rho;

    r = 1;

    opts_tmp = opts;  
    opts_tmp.k=1;
    opts_tmp.rho = param_k1.rho;
    opts_tmp.Phi = param_k1.Phi;

    prox_param_k1 = prox_param;

    if opts.flag_decomp     
       prox_param_k1.W.param_decomp{1}(1:3) = opts.lambda *  ones(1,3) ;     
       prox_param_k1.W.param_decomp{2}(1:3) = opts.lambda2 *  ones(1,3) ;         
    end

    if opts.sep_pen.flag
        prox_param_k1.W.param = opts.sep_pen.lambda_set;  
        prox_param_k1.W.sep.param_set = opts.sep_pen.lambda_set;  
    else   
        prox_param_k1.W.param(1:3) = lambda *  ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization     
    end

    if prox_param_k1.W.fix_feature_flag
        if mod(iter_out,10) == 1
            prox_param_k1.W.fix_feature_flag = false;
            prox_param_k1.W.fix_feature = zeros(size(param_k1.W(:,:,r)));
        else
            prox_param_k1.W.fix_feature_flag = true;
            prox_param_k1.W.fix_feature = abs(param_k1.W(:,:,r))>eps;
        end
    else
        prox_param_k1.W.fix_feature_flag = false;
        prox_param_k1.W.fix_feature = zeros(size(param_k1.W(:,:,r)));
    end
    
    opts_tmp.prox_param = prox_param_k1;
    opts_tmp.init.W = param_k1.W;

    if opts.sep_pen.flag
        [param_k1.W,~] = apg_W_GD_without_Z_sep(X,Y,opts_tmp);
    else
        [param_k1.W,~] = apg_W_GD_without_Z(X,Y,opts_tmp);
    end



 
    
    %E-step
%     [fun_c(iter_out),rho] = estimate_fun_without_Z(X,Y,param.W,param.Phi,param.pi,Omega,opts);
    [fun_c(iter_out),rho_global,u,v] = estimate_fun_without_Z_outlier(X,Y,param.W,param.Phi,param_k1.W,param_k1.Phi,pi_outlier,pi,Omega,opts);
    pi_outlier = sum(sum(u,3),1)./(sum(sum(u,3),1)+sum(sum(v,3),1));
%     pi_outlier = sum(sum(pi_outlier,3),1);
%     pi_outlier = pi_outlier/sum(pi_outlier);
    param.rho  = u;
    param_k1.rho = sum(v,3);
    
%     [fun_c(iter_out),rho] = estimate_fun_without_Z_weighted(X,Y,param.W,param.Phi,param.pi,Omega,param.alpha,opts);
%     
%     [param.alpha] = estimate_weights_without_Z(X,Y,param.W,param.Phi,param.pi,rho,Omega,opts);
    
%     param.rho = rho;
 
    %M-step: pi
    pi = sum(rho_global,1);
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

% disp(iter_out)

W = param.W;
Phi = param.Phi;
pi = param.pi;
rho = param.rho;
W_k1 = param_k1.W;
Phi_k1 =  param_k1.Phi;
 
end