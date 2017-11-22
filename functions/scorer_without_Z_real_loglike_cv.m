function [score] = scorer_without_Z_real_loglike_cv(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts)

Omega_p = 1 - Omega_c;

Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

 
[fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_b,opts);
 
 
score = [fun_c fun_c fun_c]/3;
    

end