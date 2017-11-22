function [score] = scorer_without_Z_real_int_complete_cv_loglike(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts)

% Omega_p = 1 - Omega_c;
% 
% Omega_c = Omega_c .* Omega_b;
% Omega_p = Omega_p .* Omega_b;
% 
% if opts.k > 1
    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
%     [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_softmax_nmse_auc(X,Y,W,Phi,rho,pi,Omega_p,opts);
%  
% else
%     [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega_p,opts);
% end

 score = [fun_c,fun_c , fun_c];
    

end