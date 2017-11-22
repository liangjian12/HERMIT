function [mean_sparse_rate,share,mean_spec_rate,spec_rate,sparse_rate] = eval_W(W)

[d,m,k] = size(W);


W = double(abs(W) > eps);

sparse  = squeeze(sum(sum(W,1),2));
sparse_rate =sparse /(d*m);
mean_sparse_rate = mean(sparse_rate);
sparse = sparse(:);

a = prod(W,3);
b = sum(W,3);bb=b;
b = double(b>0);
share = sum(a(:))/max(1,sum(b(:)));

[row,col] = find(bb==1);
spec = zeros(k,1);
for i = 1:length(row)
   
    ii = row(i);
    jj = col(i);
    
    a = W(ii,jj,:);
    a = a(:);
    idx = find(a);
    spec(idx) = spec(idx) + 1;
    
    
end
spec_rate = spec./sparse;
mean_spec_rate = mean(spec_rate);
 
end