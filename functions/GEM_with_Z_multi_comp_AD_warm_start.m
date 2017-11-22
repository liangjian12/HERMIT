function [rho,pi,W,Z,Phi,fun_c] = GEM_with_Z_multi_comp_AD_warm_start(X,Y,k,lambda,tau,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

% the default setting of regularization
if isempty(prox_param)  
    prox_param.W.method = 'L1';  % regularization method, options are : 'L2' for L2 norm; 'L1' for L1 norm;  'EN' for Elastic Net;  'GS' for Group Sparsity, which is only valid for matrix;     
    prox_param.Z.method = 'GS_row';
end
opts.prox_param = prox_param;

if isempty(opts.sep_pen.lambda_set)
    opts.sep_pen.lambda_set = opts.lambda * ones(1,3);
end
if isempty(opts.sep_pen.tau_set)
    opts.sep_pen.tau_set =  opts.tau * ones(1,3);
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
opts.rho = rho; 

%根据是否采用warm start，选择算法外环迭代次数
if opts.warmFlag
    warm_start_step=5;
else
    warm_start_step=1;
end
 
final_lambda = lambda;
final_lambda_set = opts.sep_pen.lambda_set;
firstLambdaFactor = 0.8*opts.max_lambda / final_lambda;
firstLambdaFactor_set = 0.8*opts.max_lambda ./ final_lambda_set;

final_tau = tau;
final_tau_set = opts.sep_pen.tau_set;
firstTauFactor = 0.8*opts.max_tau / final_tau;
firstTauFactor_set = 0.8*opts.max_tau ./ final_tau_set;

% firsttauFactor =  100;
% firsttauFactor_set =   100*ones(1,3);
if warm_start_step>1
    cont_factors_lambda = 10.^[log10(firstLambdaFactor):log10(1/firstLambdaFactor)/(warm_start_step-1):0]';
    cont_factors_lambda_set = ones(warm_start_step,3);
    for i_type = 1:3
        if length( 10.^[log10(firstLambdaFactor_set(i_type)):log10(1/firstLambdaFactor_set(i_type))/(warm_start_step-1):0]') == warm_start_step
        cont_factors_lambda_set(:,i_type) = 10.^[log10(firstLambdaFactor_set(i_type)):log10(1/firstLambdaFactor_set(i_type))/(warm_start_step-1):0]';
        end
        
    end
else
    cont_factors_lambda=1;
    cont_factors_lambda_set = ones(warm_start_step,3);
end

if warm_start_step>1
    cont_factors_tau = 10.^[log10(firstTauFactor):log10(1/firstTauFactor)/(warm_start_step-1):0]';
    cont_factors_tau_set = ones(warm_start_step,3);
    for i_type = 1:3
        if length( 10.^[log10(firstTauFactor_set(i_type)):log10(1/firstTauFactor_set(i_type))/(warm_start_step-1):0]') == warm_start_step
        cont_factors_tau_set(:,i_type) = 10.^[log10(firstTauFactor_set(i_type)):log10(1/firstTauFactor_set(i_type))/(warm_start_step-1):0]';
        end
        
    end
else
    cont_factors_tau=1;
    cont_factors_tau_set = ones(warm_start_step,3);
end
 
%算法开始 
 
for warm_loop=1:warm_start_step      %算法外环，warm start的循环
         

    lambda = final_lambda * cont_factors_lambda(warm_loop);
    opts.sep_pen.lambda_set = final_lambda_set .* cont_factors_lambda_set(warm_loop,:);
    tau = final_tau * cont_factors_tau(warm_loop);
    opts.sep_pen.tau_set = final_tau_set .* cont_factors_tau_set(warm_loop,:);
    
    if warm_loop == warm_start_step       %warm start前期约束宽松，最后给予标准约束                
       opts.stop_param.fun  = 2.5e-5*m*k;
       opts.stop_param.param = 2.5e-2*m*k;
       opts.EPS = 1e-6;
    else 
       opts.stop_param.fun  = 2.5e-4*m*k;
       opts.stop_param.param = 2.5e-1*m*k;
       opts.EPS = 1e-3;
    end

 
for iter_out = 1: opts.maxIter_out
    
   opts.iter_out_gem = iter_out;
    
   param_pre = param;
    
    %M-step: parameter 
      
    opts.rho = param.rho;
    opts.Phi = param.Phi;
    prox_param.W.param = lambda .* param.pi.^(gamma) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
    prox_param.Z.param = tau .* param.pi.^(gamma) ; %k

    if opts.sep_pen.flag
        prox_param.W.sep.param_set = opts.sep_pen.lambda_set(:) * param.pi.^(gamma) ; %3,k 
        prox_param.Z.sep.param_set = opts.sep_pen.tau_set(:)    * param.pi.^(gamma) ; %3,k 
    end
    
    if opts.sep_pen_inside.W.flag
        prox_param.W.param = opts.sep_pen.lambda_set(:) * param.pi.^(gamma) ; %3,k 
    end
    
    if opts.sep_pen_inside.Z.flag
        prox_param.Z.param = opts.sep_pen.tau_set(:)    * param.pi.^(gamma) ; %3,k 
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

    if prox_param.Z.fix_feature_flag
        if mod(iter_out,10) == 1
            prox_param.Z.fix_feature_flag = false;
            prox_param.Z.fix_feature = zeros(n,m,k);
        else
            prox_param.Z.fix_feature_flag = true;
            prox_param.Z.fix_feature = abs(param.Z_cube)>eps;
        end
    else
        prox_param.Z.fix_feature_flag = false;
        prox_param.Z.fix_feature = zeros(n,m,k);
    end
    
    opts.prox_param = prox_param;
    
    opts.init.W = W;
    opts.init.Z = Z;
    
    
        
        
    opts.fix.Z = param.Z;    
    if opts.sep_pen.flag
        [param.W] = apg_W_GD_only_W_multi_comp_sep(X,Y,opts);
    else
        opts.fix.Z = param.Z;
        [param.W] = apg_W_GD_only_W_multi_comp(X,Y,opts);
    end
    
    opts.init.W = param.W;
    if opts.sep_pen_outside.Z.flag  
        [param.Z] = apg_W_GD_only_Z_multi_comp_sep(X,Y,opts);
    else
        [param.Z] = apg_W_GD_only_Z_multi_comp(X,Y,opts);
    end

%     [param.W,param.Z,param.Phi,param.W_cube,param.Z_cube] = apg_W_GD_with_Z_multi_comp(X,Y,opts);
    
     
 
    %E-step
    [fun_c(iter_out),rho] = estimate_fun_with_Z(X,Y,param.W,param.Z,param.Phi,param.pi,Omega,opts);
    
    param.rho = rho;
 
 
    %M-step: pi
    pi = sum(rho,1);
    pi = pi/sum(pi);
    
    
%     if gamma>0
%         eta = 0.1;        
%         for r = 1:k
%             tmp = abs(param.W(:,:,r));
%             L1_W(r) = sum(tmp(:));
%             tmp = abs(param.Z(:,:,r));
%             L1_Z(r) = sum(tmp(:));
%         end        
%         L_pre = loss_pi_with_Z(rho,param_pre.pi,gamma,lambda,L1_W,tau,L1_Z);
%         L = L_pre;
%         
%         cont = 1;
%         while L>=L_pre && cont < 10;        
%             pi_new = param_pre.pi + eta^cont * (pi - param_pre.pi);
%             L = loss_pi_with_Z(rho,pi_new,gamma,lambda,L1_W,tau,L1_Z);
%             cont = cont + 1;
%         end
%         
%         pi = pi_new;
%     end        
    
    param.pi = pi;
    
     if iter_out>1
        d1 = param_dist_without_Z(param_pre,param,'max');
        d2 = abs(fun_c(iter_out)-fun_c(iter_out-1))/(1+abs(fun_c(iter_out)));
        if d2 <= opts.stop_param.fun || d1 <= opts.stop_param.param
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
end