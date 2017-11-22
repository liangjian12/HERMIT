function [fun_c,nmi_val,PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = scorer_with_Z_all_test_prob(X,Y,Z_idx,W,Phi,pi,Omega,opts)

    [fun_c,Z_idx_out,rho] = eval_with_Z_prob(X,Y,W,Phi,pi,Omega,opts);
    Z_idx = Z_idx.*Omega;
    [PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z] = eval_param_with_Z_prob(W,opts.true_W,opts.d_total,Z_idx_out,Z_idx);
    [~,label_out] = max(rho,[],2);
    nmi_val = nmi(label_out,opts.label_true);
    
end