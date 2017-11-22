function rho = softmax2onehot(rho)


[~,idx_max] = max(rho,[],2);
rho = full(sparse([1:size(rho,1)]',idx_max,1));
            

end