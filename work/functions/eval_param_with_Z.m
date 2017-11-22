function [PR_W,RC_W,FP_W,AUC_W,l2_norm_W,PR_Z,RC_Z,FP_Z,AUC_Z,l2_norm_Z] = eval_param_with_Z(W,W_true,d_total,Z_idx_out,Z_idx,Y_outlier_out,Y_outlier,Omega)
 

w_true = [];
w_out = [];
k = length(W);

for r = 1:k
   tmp_true =  W_true{r}(1:d_total,:);
   tmp_true = bsxfun(@rdivide,tmp_true,tmp_true(1,:));
   tmp_out = W{r};
   tmp_out = bsxfun(@rdivide,tmp_out,tmp_out(1,:));
   w_true = [w_true;tmp_true(:)';];
   w_out = [w_out;tmp_out(:)';];
end
 
 
[idx,w_out] = greedy_match(w_true,w_out);

w_true = w_true(:);
w_out = w_out(:);

l2_norm = w_true - w_out;
l2_norm_W = sum(l2_norm.*l2_norm).^0.5;

w_out = abs(w_out) > eps;
w_true = abs(w_true) > eps;

 
PR_W = sum(w_out == true & w_true == true)/max(1,sum(w_out));
RC_W = sum(w_out == true & w_true == true)/max(1,sum(w_true));
FP_W = sum(w_out == true & w_true == false)/max(1,sum(~w_true));

AUC_W = scoreAUC(w_true,w_out);

 

l2_norm = Y_outlier(:) - Y_outlier_out(:);
 
l2_norm_Z = sum(l2_norm.*l2_norm).^0.5;

z_idx_out = Z_idx_out(:) > 0;
z_idx = Z_idx(:)>0;

PR_Z = sum(z_idx_out == true & z_idx == true)/max(1,sum(z_idx_out));
RC_Z = sum(z_idx_out == true & z_idx == true)/max(1,sum(z_idx));
FP_Z = sum(z_idx_out == true & z_idx == false)/max(1,sum(~z_idx));

AUC_Z = scoreAUC(z_idx,z_idx_out);

end