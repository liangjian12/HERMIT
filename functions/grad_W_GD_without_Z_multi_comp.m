function dx = grad_W_GD_without_Z_multi_comp(x, opts)

 
X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
k = opts.k; 
 
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = zeros(d,m,k);
for r = 1:k
    W(:,:,r) = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
end
 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;

dW = zeros(size(W)); 

 
for r = 1:k
    

for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    [dW(:,idx,r)] = grad_exp_rho_without_Z(X,Y(:,idx),W(:,idx,r),Phi(idx,idx,r),Omega(:,idx),rho(:,r),opts.task_type_name{label_type});
end
 
  
end

dx = [dW(:);];

dx =  - dx  ; 



end