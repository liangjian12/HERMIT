function [Y_pred,Omega_p,rho,fun_c] = FMR_predict(X,Y,W,Phi,pi,pi_each,Omega_c,Omega_b,opts,method)

Omega_c_save = Omega_c;
Omega_p = 1 - Omega_c;
Omega_c = Omega_c .* Omega_b;
Omega_p = Omega_p .* Omega_b;

if opts.k > 1
    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
 
%     [nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_kn_softmax_nmse_auc(X,Y,W,Phi,rho,pi,Omega_p,opts);
    [Y_pred] = FMR_predict_softmax_giv_rho(X,W,Phi,rho,pi,Omega_p,opts);
     
else
    [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega_b,opts);
    rho = 1;
    [Y_pred] =  FMR_predict_k1(X,W,Phi,pi,Omega_p,opts);
end

% if strcmp(method,'SEP')
%     
%     [fun_c,nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_SEP_complete(X,Y,W,Phi,pi,pi_each,Omega_c_save,Omega_b,opts);
%     fun_c = sum(fun_c);
%     
%  
% end
    

end