function v = soft_thresh_W_GD_with_Z(x, t, opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);

% re-organize the parameters into the forms of (W,Z) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 

if opts.max_norm_regu.flag
idx_poiss = opts.task_type ==3;
if sum(idx_poiss)>0 
    W_poiss = W(:,idx_poiss);
    
    W_poiss_1 = W_poiss(1,:);
    W_poiss_rest = W_poiss(2:end,:);
    
    max_norm = max(abs(W_poiss_1(:)));
    th  = opts.max_norm_regu.W.param(1);
    if max_norm > th
        W_poiss_1(abs(W_poiss_1)>th) = W_poiss_1(abs(W_poiss_1)>th)/(2*max_norm);
    end
    
    max_norm = max(abs(W_poiss_rest(:)));
    th  = opts.max_norm_regu.W.param(2);
    if max_norm > th
        W_poiss_rest(abs(W_poiss_rest)>th) = W_poiss_rest(abs(W_poiss_rest)>th)/(2*max_norm);
    end
    
    W_poiss(1,:) = W_poiss_1;
    W_poiss(2:end,:) = W_poiss_rest;
 
    W(:,idx_poiss) = W_poiss;
    
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
W(2:end,:) = sub_prox_fun(W(2:end,:),t,opts.prox_param.W.method,opts.prox_param.W.param,...
    opts.prox_param.W.fix_feature_flag,opts.prox_param.W.fix_feature(2:end,:));
Z = sub_prox_fun(Z,t,opts.prox_param.Z.method,opts.prox_param.Z.param,...
    opts.prox_param.Z.fix_feature_flag,opts.prox_param.Z.fix_feature);

 
 
% re-organize the parameters into the parameter vector x.
v = [W(:);Z(:);x];


end