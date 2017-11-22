function [auc] = approx_AUC_sub_k(Y,mu,pi)

n = length(Y);
k = length(pi);
idx_pos = Y == 1;
idx_neg = Y == 0;
Y_pos = Y(idx_pos);
Y_neg = Y(idx_neg);
mu_pos = mu(idx_pos,:);
mu_neg = mu(idx_neg,:);
n_pos = sum(idx_pos);
n_neg = sum(idx_neg);
mat = zeros(n_pos,n_neg);
for i = 1:n_pos
    for j = 1:n_neg
        A = zeros(k);
        w = zeros(k);
        for ii = 1:k
            for jj = 1:k
                A(ii,jj) =  double(mu_pos(i,ii)>mu_neg(j,jj)) ;
                w(ii,jj) =  pi(ii) * pi(jj);
            end
        end
        mat(i,j) = weighted_vote(A(:)',w(:)');
    end
end
% mat = double(mat);
auc = sum(mat(:))/(n_pos*n_neg);

end