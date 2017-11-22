function [score] = scorer_without_Z_real_GL_cv_giv_rho(X,Y,W,Phi,pi,pi_each,Omega,opts,rho)

flag_use_loglikelihood = false;
if flag_use_loglikelihood
    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
    score = [fun_c, fun_c ,fun_c]/3;
else 
    if opts.k > 1
        [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_softmax_nmse_auc(X,Y,W,Phi,rho,pi,Omega,opts);
    else
        [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega,opts);
    end
    score = [- nmse, auc , -nmse_poiss];

end


end