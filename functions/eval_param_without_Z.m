function [PR,RC,FP,AUC,l2_norm,idx] = eval_param_without_Z(W,W_true,d_total)
 

w_true = [];
w_out = [];
k = size(W,3);

for r = 1:k
   tmp_true =  W_true(1:d_total,:,r);
   tmp_true = bsxfun(@rdivide,tmp_true,tmp_true(1,:));
   tmp_out = W(:,:,r);
   tmp_out = bsxfun(@rdivide,tmp_out,tmp_out(1,:));
   w_true = [w_true;tmp_true(:)';];
   
   w_out = [w_out;tmp_out(:)';];
end


 
[idx,w_out] = greedy_match(w_true,w_out);

w_true = w_true(:);
w_out = w_out(:);

l2_norm = w_true - w_out;
l2_norm = sum(l2_norm.*l2_norm).^0.5;

w_out_binary = abs(w_out) > eps;
w_true = abs(w_true) > eps;

 
PR = sum(w_out_binary == true & w_true == true)/sum(w_out_binary);
RC = sum(w_out_binary == true & w_true == true)/sum(w_true);
FP = sum(w_out_binary == true & w_true == false)/sum(~w_true);

[~,~,~,AUC] = perfcurve(w_true,abs(w_out),1);
 
 
 

end