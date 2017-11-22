function opts = opti_close_Phi_with_Z_multi_comp(x,opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
 
numel_loss = numel(Y);  % number of all targets
k = opts.k;
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = zeros(d,m,k);
Z = zeros(n,m,k);
for r = 1:k
    W(:,:,r) = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
end
for r = 1:k
    Z(:,:,r) = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 
end


Phi = opts.Phi; 
Omega = opts.Omega;
rho  = opts.rho;
 
for r = 1:k
 
for label_type = 1:1
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    
    if label_type == 1
        X_tmp = bsxfun(@times,X,sqrt(rho(:,r)));
        Y_tmp = bsxfun(@times,Y,sqrt(rho(:,r)));
    else
        X_tmp = X;
        Y_tmp = Y;
    end
    
    G = X_tmp*W(:,idx,r) + Z(:,idx,r);
    
    [Phi{r}(idx,idx)] = link_fun_phi_rho(Y_tmp(:,idx),G,Omega(:,idx),rho(:,r),opts.task_type_name{label_type});
 
end

end
 
opts.Phi = Phi;


end