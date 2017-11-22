function [mean_sparse_rate,share,mean_spec_rate,nmse,auc,aupr,nmse_poiss] = scorer_without_Z_real_with_pred(Y_true,Y_pred,W,Omega_p,opts,method)
 
% if opts.k > 1
% %     [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_c,opts);
%     [mean_sparse_rate,share,mean_spec_rate] = eval_W(W);
%     [nmse,auc,aupr,nmse_poiss] = score_with_pred(Y_true,Y_pred,Omega_p,opts);
%     %[ fun_c ] = estimate_fun_giv_rho_without_Z(X,Y,W,Phi,pi,rho,Omega_p,opts);
% %     [fun_c] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega_b,opts);
%     
% else
%     [fun_c] = estimate_fun_without_Z_k1(X,Y,W,Phi,pi,Omega_b,opts);
    [mean_sparse_rate] = eval_W_k1(W);
    share=1;
    mean_spec_rate=1;
    [nmse,auc,aupr,nmse_poiss] = score_with_pred_k1(Y_true,Y_pred,Omega_p,opts);
% end

% if strcmp(method,'SEP')
%     
%     [fun_c,nmse,auc,aupr,nmse_poiss] = estimate_fun_without_Z_SEP_complete(X,Y,W,Phi,pi,pi_each,Omega_c_save,Omega_b,opts);
%     fun_c = sum(fun_c);
%     
%  
% end
    

end