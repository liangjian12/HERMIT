function rho = rho2onehot(rho)
n = size(rho,1);
[~,idx_max] = max(rho,[],2);
rho = full(sparse([1:n]',idx_max,1));

end