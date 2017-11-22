function v = soft_thresh_W_GD(x, t, opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);

% re-organize the parameters into the forms of (W,Z) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = [];

% soft_thresh for each parameter
W = sub_prox_fun(W,t,opts.prox_param.W.method,opts.prox_param.W.param);
Z = sub_prox_fun(Z,t,opts.prox_param.Z.method,opts.prox_param.Z.param);
 
 
% re-organize the parameters into the parameter vector x.
v = [W(:);Z(:);x];


end