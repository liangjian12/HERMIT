function [PR,RC,FP,AUC] = eval_param_union_without_Z_k1(W,W_true,d_total)
 

w_true = [];
w_out = [];


for r = 1:size(W,3)
   tmp_out = W(:,:,r);
   tmp_out = bsxfun(@rdivide,tmp_out,tmp_out(1,:));
   w_out = [w_out;tmp_out(:)';];
end

for r = 1:size(W_true,3)
   tmp_true =  W_true(1:d_total,:,r);
   tmp_true = bsxfun(@rdivide,tmp_true,tmp_true(1,:));
   w_true = [w_true;tmp_true(:)';];
end
 
w_true = sum(abs(w_true),1);
w_out = sum(abs(w_out),1);

w_true = w_true(:);
w_out = w_out(:);
 
w_out_binary = abs(w_out) > eps;
w_true = abs(w_true) > eps;

 
PR = sum(w_out_binary == true & w_true == true)/sum(w_out_binary);
RC = sum(w_out_binary == true & w_true == true)/sum(w_true);
FP = sum(w_out_binary == true & w_true == false)/sum(~w_true);

[~,~,~,AUC] = perfcurve(w_true,abs(w_out),1);
 
 
 

end