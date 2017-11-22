function [rho,pi,W,Z,Phi] = GEM_with_Z_k1_multi_comp(X,Y,k,lambda,tau,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

rho = ones(n,1);
pi = 1;

opts.iter_out_gem = 1; 
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
param.Phi = Phi;
param.Z = Z; 
param.rho = rho;
 
r = 1;
  
opts_tmp = opts;  
opts_tmp.rho = param.rho(:,r);
opts_tmp.Phi = param.Phi(:,:,r);

prox_param.W.param = lambda .* param.pi.^(gamma) ; % coefficient of regularization, e.g., in ( lambda ||x||_2^2 ), lambda is the coefficient of regularization        
prox_param.Z.param = tau .* param.pi.^(gamma) ; %k

if opts.sep_pen.flag
    prox_param.W.sep.param_set = opts.sep_pen.lambda_set(:) * param.pi.^(gamma) ; %3,k 
    prox_param.Z.sep.param_set = opts.sep_pen.tau_set(:)    * param.pi.^(gamma) ; %3,k 
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

opts_tmp.prox_param = prox_param;

opts_tmp.init.W = W;
opts_tmp.init.Z = Z;

if opts.sep_pen.flag
    [W(:,:,1),Z(:,:,1),Phi(:,:,1)] = apg_W_GD_with_Z_multi_comp_sep(X,Y,opts);
else
    [W(:,:,1),Z(:,:,1),Phi(:,:,1)] = apg_W_GD_with_Z_multi_comp(X,Y,opts);
end

end
 
 
 