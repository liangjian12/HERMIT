function [fun_c,mean_sparse_rate,share,mean_spec_rate,nmse,auc,aupr,nmse_poiss,rho] = scorer_without_Z_real_GL_k1(X,Y,W,Phi,pi,pi_each,Omega ,opts,method)

 
    [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega ,opts);
    [mean_sparse_rate] = eval_W_k1(W);
    share=1;
    mean_spec_rate=1;
    [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega ,opts);
 
    

end