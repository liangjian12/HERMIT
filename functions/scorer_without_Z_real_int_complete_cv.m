function [score] = scorer_without_Z_real_int_complete_cv(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts)

Omega_p = 1 - Omega_c;

Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

flag_use_loglikelihood = false;
if flag_use_loglikelihood
    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
    score = [fun_c, fun_c, fun_c]/3;
else
    if opts.k > 1
        [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
        [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_softmax_nmse_auc(X,Y,W,Phi,rho,pi,Omega_p,opts);

    else
        [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
        [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega_p,opts);
    end

    score = [- nmse, auc , -nmse_poiss];
end
    

end