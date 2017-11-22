function [PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = eval_param_only_Z(Z_idx_out,Z_idx,Y_outlier_out,Y_outlier)
 
l2_norm = Y_outlier(:) - Y_outlier_out(:);
 
l2_norm_Z = sum(l2_norm.*l2_norm).^0.5;

z_idx_out = Z_idx_out(:) > 0;
z_idx = Z_idx(:)>0;

PR_Z = sum(z_idx_out == true & z_idx == true)/max(1,sum(z_idx_out));
RC_Z = sum(z_idx_out == true & z_idx == true)/max(1,sum(z_idx));
FP_Z = sum(z_idx_out == true & z_idx == false)/max(1,sum(~z_idx));

AUC_Z = scoreAUC(z_idx,z_idx_out);

end