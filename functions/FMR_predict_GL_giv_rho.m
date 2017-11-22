function [Y_pred,Omega ,rho ] = FMR_predict_GL_giv_rho(X,Y,W,Phi,pi,pi_each,Omega ,opts,method,rho)

 
if opts.k > 1
 
    [Y_pred] = FMR_predict_softmax_giv_rho(X,W,Phi,rho,pi,Omega,opts);
     
else
%     [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega_b,opts);
%     rho = 1;
    [Y_pred] =  FMR_predict_k1(X,W,Phi,pi,Omega,opts);
end

% if strcmp(method,'SEP')
%     
%     [fun_c,nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_SEP_complete(X,Y,W,Phi,pi,pi_each,Omega_c_save,Omega_b,opts);
%     fun_c = sum(fun_c);
%     
%  
% end
    

end