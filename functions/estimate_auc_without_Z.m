function auc = estimate_auc_without_Z(X,Y,W,Phi,pi,Omega,opts)

d_total = opts.d_total;
W_true = opts.true_W;

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

[~,w_out] = greedy_match(w_true,w_out);

w_true = w_true(:);
w_out = w_out(:);


w_out = abs(w_out) > eps;
w_true = abs(w_true) > eps;
 
auc = scoreAUC(w_true,w_out);
 

end