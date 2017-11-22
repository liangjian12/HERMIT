function [W,Phi,pi,W_k1,Phi_k1,pi_outlier] = solver_MIX_without_Z_outlier(X,Y,Omega,k,opts,prox_param)

opts.initial_scale = opts.initial_scale_scale/size(Y,2); 
opts.Omega = Omega;

   
[W,Phi,pi,W_k1,Phi_k1,pi_outlier] = solver_sub_without_Z_outlier(X,Y,k,opts,prox_param);
    




end