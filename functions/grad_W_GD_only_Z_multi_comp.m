function dx = grad_W_GD_only_Z_multi_comp(x, opts)

 
X = opts.X;
Y = opts.Y;

W = opts.fix.W;

[n,d] = size(X);
[n,m] = size(Y);
k = opts.k;  
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
 
Z = zeros(n,m,k);
for r = 1:k
    Z(:,:,r) = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 
end
 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;
 
dZ = zeros(size(Z));
 
for r = 1:k
    
  
for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    [~,dZ(:,idx,r)] = grad_exp_rho_with_Z(X,Y(:,idx),W(:,idx,r),Z(:,idx,r),Phi(idx,idx,r),Omega(:,idx),rho(:,r),opts.task_type_name{label_type});
end
 
  
end
dx = [dZ(:);];
dx =  - dx  ; 



end