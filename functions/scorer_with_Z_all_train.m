function [fun_c,nmi_val,PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = scorer_with_Z_all_train(X,Y,Y_outlier,Z_idx,W,Z,Phi,pi,Omega,opts)

    [fun_c,rho] = estimate_fun_with_Z(X,Y,W,Z,Phi,pi,Omega,opts);
    [Z_idx_out,Y_outlier_out] = compute_outlier(X,W,Z,Phi,rho,opts);
     Y_outlier = Y_outlier.*Omega;
    [PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = eval_param_with_Z(W,opts.true_W,opts.d_total,Z_idx_out,Z_idx,Y_outlier_out,Y_outlier,Omega);
    [~,label_out] = max(rho,[],2);
    nmi_val = nmi(label_out,opts.label_true_train);
    
    

end