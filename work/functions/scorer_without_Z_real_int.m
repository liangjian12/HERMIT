function [fun_c,mean_sparse_rate,share,mean_spec_rate,nmse,auc] = scorer_without_Z_real_int(X,Y,W,Phi,pi,pi_each,Omega,opts,method)


if opts.k > 1
    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega,opts);
    [mean_sparse_rate,share,mean_spec_rate] = eval_W(W);
    [nmse,auc] = estimate_fun_without_Z_kn_nmse_auc(X,Y,W,Phi,rho,pi,Omega,opts);
else
    [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega,opts);
    [mean_sparse_rate] = eval_W_k1(W);
    share=1;
    mean_spec_rate=1;
    [nmse,auc] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega,opts);
end

if strcmp(method,'SEP')
    
    [fun_c] = estimate_fun_without_Z_SEP(X,Y,W,Phi,pi_each,Omega,opts);
    fun_c = sum(fun_c);
    
elseif strcmp(method,'MAR')
    
    [fun_c] = estimate_fun_without_Z_MAR(X,Y,W,Phi,pi_each,Omega,opts);
    fun_c = sum(fun_c);
    
end
    

end