function [score] = scorer_only_Z_real_int_complete_cv(X,Y,W,Z,Phi,pi,pi_each,Omega_c,Omega_b,opts)

Omega_p = 1 - Omega_c;

Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

if opts.k > 1
    [fun_c,rho] = estimate_fun_with_Z(X,Y,W,Z,Phi,pi,Omega_c,opts);
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_only_Z_kn_softmax_nmse_auc(X,Y,W,Z,Phi,rho,pi,Omega_p,opts);
 
else
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_only_Z_k1_nmse_auc(X,Y,W,Z,Phi,pi,Omega_p,opts);
end

 score = [- nmse, auc , -nmse_poiss];
    

end