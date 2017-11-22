function [rho,pi,W,Phi] = GEM_without_Z_k1(X,Y,k,lambda,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

rho = ones(n,1);
pi = 1;

opts.iter_out_gem = 1; 
% the default setting of regularization
if isempty(prox_param)  
    prox_param.W.method = 'L1';  % regularization method, options are : 'L2' for L2 norm; 'L1' for L1 norm;  'EN' for Elastic Net;  'GS' for Group Sparsity, which is only valid for matrix;     
end
opts.prox_param = prox_param;

if isempty(opts.sep_pen.lambda_set)
    opts.sep_pen.lambda_set = opts.lambda * ones(1,3);
end
 


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

iter_num = 1;
if isfield( opts.prox_param.W, 'capL1_flag')
    if opts.prox_param.W.capL1_flag
        iter_num = 50;
    end
end

for iter_out = 1:iter_num

param.W = W;
param.Phi = Phi;
param.pi = 1; 
param.rho = rho;
 
r = 1;
  
opts_tmp = opts;  
opts_tmp.rho = param.rho(:,r);
opts_tmp.Phi = param.Phi(:,:,r);
   

if opts.flag_decomp     
   prox_param.W.param_decomp{1}(1:3) = opts.lambda *  ones(1,3) ;     
   prox_param.W.param_decomp{2}(1:3) = opts.lambda2 *  ones(1,3) ;         
end

if opts.sep_pen.flag
    prox_param.W.param = opts.sep_pen.lambda_set;  
    prox_param.W.sep.param_set = opts.sep_pen.lambda_set;  
else   
    prox_param.W.param(1:3) = lambda *  ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization     
end

if prox_param.W.fix_feature_flag    
    prox_param.W.fix_feature_flag = false;
    prox_param.W.fix_feature = zeros(size(param.W(:,:,r)));
else
    prox_param.W.fix_feature_flag = false;
    prox_param.W.fix_feature = zeros(size(param.W(:,:,r)));
end

if isfield(opts.prox_param.W,'ada_lasso_flag')  
    if opts.prox_param.W.ada_lasso_flag 
    prox_param.W.ada_lasso_weight_this = prox_param.W.ada_lasso_weight{r};
    end
end

if isfield( opts.prox_param.W, 'capL1_flag')
    if opts.prox_param.W.capL1_flag
         if iter_out == 1
            prox_param.W.capL1_weight_this = ones(size(param.W(:,:,r)));
         else
            prox_param.W.capL1_weight_this = repmat(sum(abs(param.W(:,:,r)),2) < opts.theta,1,m); 
         end
    end
end

opts_tmp.prox_param = prox_param;
opts_tmp.init.W = param.W(:,:,r);

if opts.sep_pen.flag
    [W(:,:,1),~] = apg_W_GD_without_Z_sep(X,Y,opts_tmp);
else
    [W(:,:,1),~] = apg_W_GD_without_Z(X,Y,opts_tmp);
end

[fun_c(iter_out),rho] = estimate_fun_without_Z(X,Y,W,param.Phi,param.pi,Omega,opts);
% d2 = abs(fun_c(iter_out)-fun_c(iter_out-1))/(1+abs(fun_c(iter_out)));
if iter_out>1
%         d1 = param_dist_without_Z(param_pre,param,'max');
    d2 = abs(fun_c(iter_out)-fun_c(iter_out-1))/(1+abs(fun_c(iter_out)));
    if d2 <= opts.stop_param.fun  
        break
    end
end

end

end
 
 
 