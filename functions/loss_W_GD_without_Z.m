function L = loss_W_GD_without_Z(x, opts)

 
X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
 
 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Phi = opts.Phi; 
Omega = opts.Omega;
rho = opts.rho;


L = compute_loss_r_without_Z(X,Y,W,Phi,rho,Omega,opts);
L = -L;



end