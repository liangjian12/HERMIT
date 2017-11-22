function [fun_c,Z_idx,rho,Omega] = eval_with_Z_prob(X,Y,W,Phi,pi,Omega,opts)
 

 [Z_idx] = compute_Z_idx(X,Y,W,Phi,Omega,opts);
 
Omega = Omega.*(~Z_idx);

[fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega,opts);



end