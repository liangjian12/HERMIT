function [rho,Z,fun] = GEM_only_Z_k1(X,Y,k,tau,gamma,prox_param,opts)

[n,d] = size(X);
[n,m] = size(Y);

Omega = opts.Omega;

rho = ones(n,1);

pi = opts.fix.pi;

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

r = 1; 
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
 
%E-step
[fun,rho] = estimate_fun_only_Z(X,Y,param.W,param.Z,param.Phi,param.pi,Omega,opts);

param.rho = rho; 
Z = param.Z;
 
 
rho = param.rho;
 
end