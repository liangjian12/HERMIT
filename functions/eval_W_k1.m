function [mean_sparse_rate,share,mean_spec_rate] = eval_W_k1(W)

[d,m,k] = size(W);
W = double(abs(W) > eps);

sparse  = squeeze(sum(sum(W,1),2));
sparse_rate =sparse /(d*m);
mean_sparse_rate = mean(sparse_rate);
 
 
share = 1;
mean_spec_rate = 1;
 
end