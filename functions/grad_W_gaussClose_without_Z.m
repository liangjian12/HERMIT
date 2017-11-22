function dx = grad_W_gaussClose_without_Z(x, opts)

 
X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
 
m = m - opts.task_num_each_type(1);
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;

Omega_non_Gauss = [];


dW = [];

 
for label_type = 2:3
    idx = opts.task_type == label_type;
    if sum(idx) == 0
        continue
    end
    try
    dW_tmp = grad_exp_rho_without_Z(X,Y(:,idx),W(:,idx),Phi(idx,idx),Omega(:,idx),rho,opts.task_type_name{label_type});
    catch
        disp('')
    end
    dW = [dW dW_tmp];
    Omega_non_Gauss = [Omega_non_Gauss Omega(:,idx)];
end
 

dx = [dW(:)];
numel_loss = sum(Omega_non_Gauss(:));  % number of all visible targets
dx =  - dx /(2* numel_loss); 



end