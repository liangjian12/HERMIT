function [fun,rho] = estimate_fun_without_Z_complete(X,Y,W,Phi,pi,Omega_c,opts)
% 
Omega_p = 1 - Omega_c;

[~,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
[ fun] = estimate_fun_giv_rho_without_Z(X,Y,W,Phi,pi,rho,Omega_p,opts);
     

end