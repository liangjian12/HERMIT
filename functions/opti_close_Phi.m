function opts = opti_close_Phi(x,opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
 
numel_loss = numel(Y);  % number of all targets
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = [];
Phi = eye(m); 
Omega = opts.Omega;
rho = opts.rho;

 
for label_type = 1:1
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    
    G = X*W(:,idx) + Z(:,idx);
    
    [Phi(idx,idx)] = link_fun_phi_rho(Y(:,idx),G,Omega(:,idx),rho,opts.task_type_name{label_type});
 
end
 
opts.Phi = Phi;


end