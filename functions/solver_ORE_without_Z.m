function [W,Phi,pi] = solver_ORE_without_Z(X,Y,Omega,k,opts,prox_param)


opts.Omega = Omega;
 
prox_param.W.method = 'GS';

opts.initial_scale = opts.initial_scale_scale/size(Y,2);
   
[W,Phi,pi] = solver_sub_without_Z(X,Y,k,opts,prox_param);
    




end