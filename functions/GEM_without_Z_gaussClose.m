function [rho,pi,W,Phi,fun_c] = GEM_without_Z(X,Y,k,lambda,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

% rho = exprnd(1,n,k);
% rho = bsxfun(@rdivide,rho,max(rho,[],2));
% rho(rho~=1) = 0.1;
% rho(rho==1) = 0.9;

rho = rand(n,k);
rho = bsxfun(@rdivide,rho,sum(rho,2));

pi = sum(rho,1);
pi = pi/sum(pi);

opts.rho = rho;

% the default setting of regularization
if isempty(prox_param)  
    prox_param.W.method = 'L1';  % regularization method, options are : 'L2' for L2 norm; 'L1' for L1 norm;  'EN' for Elastic Net;  'GS' for Group Sparsity, which is only valid for matrix;     
end
opts.prox_param = prox_param;
 
W = {};
Phi = {};
for r = 1:k
    W{r} = rand(d,m) * opts.opts.initial_scale;  
    Phi{r} = eye(m);
%     idx = opts.task_type ==3;
%     m_3 = sum(idx);
%     
%     Phi{r}(idx,idx) = exp(-2) * eye(m_3);  
end

param.W = W;
param.Phi = Phi;
param.pi = pi;
param.rho = rho;







for iter_out = 1: opts.maxIter_out
    
   param_pre = param;
   
   
    
    %M-step: parameter 
    for r = 1:k    
        opts_tmp = opts;  
        opts_tmp.rho = param.rho(:,r);
        opts_tmp.Phi = param.Phi{r};
        prox_param.W.param = lambda * param.pi(r)^(gamma) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
        
        opts_tmp.prox_param = prox_param;
        if  prox_param.W.ada_lasso_flag
            opts_tmp.prox_param.W.ada_lasso_weight = prox_param.W.ada_lasso_weight{r};
        end
        
        if prox_param.W.fix_feature_flag
            if mod(iter_out,10) == 1
                opts_tmp.prox_param.W.fix_feature_flag = false;
                opts_tmp.prox_param.W.fix_feature = [];
            else
                opts_tmp.prox_param.W.fix_feature_flag = true;
                opts_tmp.prox_param.W.fix_feature = abs(param.W{r})>eps;
            end
        else
            opts_tmp.prox_param.W.fix_feature_flag = false;
            opts_tmp.prox_param.W.fix_feature = [];
            
        end
        
        [param.W{r},param.Phi{r}] = apg_W_GD_without_Z(X,Y,opts_tmp);
    end
 
%     for r = 1:k    
%         param.W{r} = param.W{r} + sqrt(0.01/(1+iter_out)^(0.55)) * randn(size(param.W{r}));
%     end
    
%     for r = 1:k    
%         param.W{r}(2:end,:) = param.W{r}(2:end,:).*(rand(size(param.W{r}(2:end,:)))<0.95);
% %         param.W{r}  = param.W{r} .*(rand(size(param.W{r} ))<0.95);
%     
%     end
    
    %E-step
    [fun_c(iter_out),rho] = estimate_fun_without_Z(X,Y,param.W,param.Phi,param.pi,Omega,opts);
    
    param.rho = rho;
 
    
    
    %M-step: pi
    pi = sum(rho,1);
    pi = pi/sum(pi);
    
    
    if gamma>0
        eta = 0.1;        
        for r = 1:k
            tmp = abs(param.W{r});
            L1_W(r) = sum(tmp(:));
        end        
        L_pre = loss_pi_without_Z(rho,param_pre.pi,gamma,lambda,L1_W);
        L = L_pre;
        
        cont = 1;
        while L>=L_pre && cont < 20;        
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

        if d2 <= 1e-6 && d1 <= 1e-3
            break
        end

    end
    
%     if mod(iter_out,10) == 0    
%      disp([iter_out,fun(iter_out) ,fun_c(iter_out) ])     
%     end
     
end

W = param.W;
Phi = param.Phi;
pi = param.pi;
rho = param.rho;
 
end