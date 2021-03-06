function dx = grad_W_GD_only_Z(x, opts)

 
X = opts.X;
Y = opts.Y;

W = opts.fix.W;

[n,d] = size(X);
[n,m] = size(Y);
 
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
 
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 
 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;
 
dZ = zeros(size(Z));

 
for label_type = 1:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    [~,dZ(:,idx)] = grad_exp_rho_with_Z(X,Y(:,idx),W(:,idx),Z(:,idx),Phi(idx,idx),Omega(:,idx),rho,opts.task_type_name{label_type});
end
 

dx = [dZ(:)];
dx =  - dx ; 



end