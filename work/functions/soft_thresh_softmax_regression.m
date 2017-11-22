function v = soft_thresh_softmax_regression(x, t, opts)
 
X = opts.X;
rho = opts.rho;

[n,d] = size(X);
[n,m] = size(rho);

% re-organize the parameters into the forms of (W,Z) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
 

% soft_thresh for each parameter
 
W(2:end,:) = sub_prox_fun(W(2:end,:),t,opts.prox_param.W_softmax.method,opts.prox_param.W_softmax.param,...
    opts.prox_param.W_softmax.fix_feature_flag,opts.prox_param.W_softmax.fix_feature(2:end,:));
 
 
 
% re-organize the parameters into the parameter vector x.
v = [W(:);x];

% norm_2 = sum(v.*v).^0.5;
% v = v/norm_2;



end