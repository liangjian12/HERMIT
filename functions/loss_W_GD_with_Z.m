function L = loss_W_GD_with_Z(x, opts)

 
X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
 
 
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;

[L] = compute_loss_r_with_Z(X,Y,W,Z,Phi,Omega,rho,opts);
 
L = -L;


end