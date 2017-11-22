function [fun_c,latent_acc,PR_W,RC_W,FP_W,AUC_W,l2_norm_W] = scorer_without_Z_all(X,Y,W,Phi,pi,Omega,opts)

    [fun_c,rho] = estimate_fun_without_Z(X,Y,W,Phi,pi,Omega,opts);
    [PR_W,RC_W,FP_W,AUC_W,l2_norm_W,idx] = eval_param_without_Z(W,opts.true_W,opts.d_total);
    rho = rho(:,idx);
    [~,label_out] = max(rho,[],2);
%     nmi_val = nmi(label_out,opts.label_true);
    latent_acc = mean(label_out == opts.label_true);
    
    

end