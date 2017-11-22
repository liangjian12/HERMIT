function v = soft_thresh_W_GD_only_Z(x, t, opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);

% re-organize the parameters into the forms of (W,Z) from the vector x.
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 
 
if opts.max_norm_regu.flag
idx_poiss = opts.task_type ==3;
if sum(idx_poiss)>0 
    Z_poiss = Z(:,idx_poiss); 
    
    max_norm = max(abs(Z_poiss(:)));
    th  = opts.max_norm_regu.Z.param(1);
    if max_norm > th
        Z_poiss(abs(Z_poiss)>th) = Z_poiss(abs(Z_poiss)>th)/(2*max_norm);
    end     
   
    Z(:,idx_poiss) = Z_poiss;
end
end

% soft_thresh for each parameter
Z = sub_prox_fun(Z,t,opts.prox_param.Z.method,opts.prox_param.Z.param,...\
    opts.prox_param.Z.fix_feature_flag,opts.prox_param.Z.fix_feature);

 
 
% re-organize the parameters into the parameter vector x.
v = [Z(:);x];


end