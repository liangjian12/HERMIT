function [fun_c,nmi_val,PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = scorer_with_Z_all_test(X,Y,Y_outlier,Z_idx,W,Phi,pi,Omega,opts,prox_param)

    [fun_c,Z,rho] = eval_with_Z(X,Y,W,Phi,pi,Omega,opts,prox_param);
    
    [Z_idx_out,Y_outlier_out] = compute_outlier(X,W,Z,Phi,rho,opts);
    
    [PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = eval_param_with_Z(W,opts.true_W,opts.d_total,Z_idx_out,Z_idx,Y_outlier_out,Y_outlier,Omega);
    [~,label_out] = max(rho,[],2);
    nmi_val = nmi(label_out,opts.label_true);
    
end