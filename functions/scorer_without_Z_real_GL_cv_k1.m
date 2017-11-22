function [score] = scorer_without_Z_real_GL_cv_k1(X,Y,W,Phi,pi,pi_each,Omega ,opts)

  
 [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_k1_nmse_auc(X,Y,W,Phi,pi,Omega ,opts);
 

 score = [- nmse, auc , -nmse_poiss];
    

end