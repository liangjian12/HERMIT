function [W,Phi,pi,pi_each] = solver_capL1_without_Z(X,Y,Omega,k,opts,prox_param)

opts.initial_scale = opts.initial_scale_scale/size(Y,2); 
opts.Omega = Omega;

opts.prox_param.W.capL1_flag = 1;   
prox_param.W.capL1_flag = 1;
[W,Phi,pi] = solver_sub_without_Z(X,Y,k,opts,prox_param);

pi_each = [];
    




end