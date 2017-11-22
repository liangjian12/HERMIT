function [fun_c,mean_sparse_rate,share,mean_spec_rate,nmse,auc,nmse_poiss,...
    PR_W,RC_W,FP_W,AUC_W] = scorer_without_Z_simu_int_complete_max(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts,method)

Omega_c_save = Omega_c;
Omega_p = 1 - Omega_c;
Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

if opts.k > 1
    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
    [mean_sparse_rate,share,mean_spec_rate] = eval_W(W);
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_max_nmse_auc(X,Y,W,Phi,rho,pi,Omega_p,opts);
    [ fun_c ] = estimate_fun_giv_rho_without_Z(X,Y,W,Phi,pi,rho,Omega_b,opts);
    
    [PR_W,RC_W,FP_W,AUC_W] = eval_param_union_without_Z(W,opts.true_W,opts.d_total);

else
    [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega_b,opts);
    [mean_sparse_rate] = eval_W_k1(W);
    share=1;
    mean_spec_rate=1;
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega_p,opts);
    
    [PR_W,RC_W,FP_W,AUC_W] = eval_param_union_without_Z(W,opts.true_W,opts.d_total);
end

if strcmp(method,'SEP')
    
    [fun_c,nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_SEP_complete(X,Y,W,Phi,pi,pi_each,Omega_c_save,Omega_b,opts);
    fun_c = sum(fun_c);
    
 
end
    

end