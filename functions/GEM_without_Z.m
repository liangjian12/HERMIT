function [rho,pi,W,Phi,fun_c] = GEM_without_Z(X,Y,k,lambda,gamma,prox_param,opts)

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

param.W = W;
param.Phi = Phi;
param.pi = pi;
param.rho = rho;
opts.rho = rho;
param.alpha = ones(1,m)/m;

opts.stop_param.fun  = 2.5e-5*m*k;
opts.stop_param.param = 2.5e-2*m*k;

% opts.stop_param.fun  = 2.5e-5*k;
% opts.stop_param.param = 2.5e-2*k;
 
 
fun_c = zeros(1,opts.maxIter_out);
for iter_out = 1: opts.maxIter_out
    
    opts.iter_out_gem = iter_out;
    
%     if iter_out > 1
%         opts.warm_start.flag = true;
%     end
%     
   param_pre = param;
   
    %M-step: parameter 
    for r = 1:k    
        opts_tmp = opts;  
        opts_tmp.rho = param.rho(:,r);
        opts_tmp.Phi = param.Phi(:,:,r);
        
%         if opts.sep_pen.flag  
%             prox_param.W.sep.param_set = opts.sep_pen.lambda_set  ;  
%         else
%             prox_param.W.param(1:3) = lambda *  ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
%         end
%         
%         if opts.flag_decomp
%            
%             prox_param.W.param_decomp{1}(1:3) = opts.lambda   * ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization  
%             prox_param.W.param_decomp{2}(1:3) = opts.lambda2   * ones(1,3) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization  
%              
%         end
        
        
        
    
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
        
        if isfield( opts.prox_param.W, 'ada_lasso_flag')
            if opts.prox_param.W.ada_lasso_flag
                prox_param.W.ada_lasso_weight_this = prox_param.W.ada_lasso_weight{r};
            end
        end
        
        if isfield( opts.prox_param.W, 'capL1_flag')
            if opts.prox_param.W.capL1_flag
                 if iter_out == 1
                    prox_param.W.capL1_weight_this = ones(size(param.W(:,:,r)));
                 else
                    prox_param.W.capL1_weight_this = repmat(sum(abs(param.W(:,:,r)),2) < opts.theta,1,m); %* param.pi(r)^(gamma),1,m); 
                 end
            end
        end
        
        
        opts_tmp.prox_param = prox_param;
        opts_tmp.init.W = param.W(:,:,r);
        
        
        if opts.sep_pen.flag
            [param.W(:,:,r),~] = apg_W_GD_without_Z_sep(X,Y,opts_tmp);
        else
            [param.W(:,:,r),~] = apg_W_GD_without_Z(X,Y,opts_tmp); 
        end
       
    end
 
    
    %E-step
    [fun_c(iter_out),rho] = estimate_fun_without_Z(X,Y,param.W,param.Phi,param.pi,Omega,opts);
%     [fun_c(iter_out),rho] = estimate_fun_without_Z_weighted(X,Y,param.W,param.Phi,param.pi,Omega,param.alpha,opts);
%     
%     [param.alpha] = estimate_weights_without_Z(X,Y,param.W,param.Phi,param.pi,rho,Omega,opts);
    
    param.rho = rho;
 
    %M-step: pi
    pi = sum(rho,1);
    pi = pi/sum(pi);
    
    
%     if gamma>0
%         eta = 0.1;        
%         for r = 1:k
%             tmp = abs(param.W(:,:,r));
%             L1_W(r) = sum(tmp(:));
%         end        
%         L_pre = loss_pi_without_Z(rho,param_pre.pi,gamma,lambda,L1_W);
%         L = L_pre;
%         cont = 1;
%         while L>=L_pre && cont < 10;        
%             pi_new = param_pre.pi + eta^cont * (pi - param_pre.pi);
%             L = loss_pi_without_Z(rho,pi_new,gamma,lambda,L1_W);
%             cont = cont + 1;
%         end
%         pi = pi_new;
%     end        
    
    param.pi = pi;

    if iter_out>1
        d1 = param_dist_without_Z(param_pre,param,'max');
        d2 = abs(fun_c(iter_out)-fun_c(iter_out-1))/(1+abs(fun_c(iter_out)));
        if d2 <= opts.stop_param.fun && d1 <= opts.stop_param.param
            break
        end
    end
    
%     if mod(iter_out,10) == 0    
%      disp([iter_out,fun(iter_out) ,fun_c(iter_out) ])     
%     end
     
end

% disp(iter_out)

W = param.W;
Phi = param.Phi;
pi = param.pi;
rho = param.rho;
 
end