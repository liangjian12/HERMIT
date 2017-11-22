function dx = grad_W_GD_without_Z(x, opts)

 
X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
 

 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;
 

dW = zeros(size(W));

 
for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    [dW(:,idx)] = grad_exp_rho_without_Z(X,Y(:,idx),W(:,idx),Phi(idx,idx),Omega(:,idx),rho,opts.task_type_name{label_type});
end

% n_vis = sum(Omega,1);
% dW = bsxfun(@rdivide,dW,n_vis);
 

dx = [dW(:)];
dx =  - dx; 



end