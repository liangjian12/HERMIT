function [score] = scorer_without_Z_real_k1_GL_cv(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts)

 
[nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega_b,opts);
 

 score = [- nmse, auc , -nmse_poiss];
    

end