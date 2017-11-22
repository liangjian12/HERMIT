function v = soft_thresh_W_gaussClose_without_Z_adadelta(x, t, opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);

m = m - opts.task_num_each_type(1);

% re-organize the parameters into the forms of (W,Z) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
T =  reshape(t(1:d*m),[d m]); t(1:d*m) = []; 

idx_non_Gauss = opts.task_type ~= 1;

% soft_thresh for each parameter
W(2:end,:) = sub_prox_fun(W(2:end,:),T(2:end,:),opts.prox_param.W.method,opts.prox_param.W.param,...
    opts.prox_param.W.ada_lasso_flag,opts.prox_param.W.ada_lasso_weight(2:end,idx_non_Gauss),...
    opts.prox_param.W.fix_feature_flag,opts.prox_param.W.fix_feature(2:end,idx_non_Gauss));
 
 
 
% re-organize the parameters into the parameter vector x.
v = [W(:);x];

% norm_2 = sum(v.*v).^0.5;
% v = v/norm_2;


end