function [W,Z,Phi,pi] = solver_ORE_with_Z(X,Y,Omega,k,opts,prox_param)

  
 
prox_param.W.method = 'GS';
opts.initial_scale = opts.initial_scale_scale/size(Y,2);   
opts.Omega = Omega;
[W,Z,Phi,pi] = solver_sub_with_Z(X,Y,k,opts,prox_param);
    




end