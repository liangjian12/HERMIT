function [auc] = approx_AUC_sub(Y,mu)

n = length(Y);

idx_pos = Y == 1;
idx_neg = Y == 0;
Y_pos = Y(idx_pos);
Y_neg = Y(idx_neg);
mu_pos = mu(idx_pos);
mu_neg = mu(idx_neg);
n_pos = sum(idx_pos);
n_neg = sum(idx_neg);
mat = zeros(n_pos,n_neg);
for i = 1:n_pos
    for j = 1:n_neg
        mat(i,j) = mu_pos(i) > mu_neg(j);
    end
end
mat = double(mat);
auc = sum(mat(:))/(n_pos*n_neg);

end