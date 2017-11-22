function v = soft_thresh_W_GD_only_Z_multi_comp_sep(x, t, opts)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);
k = opts.k;
% re-organize the parameters into the forms of (W,Z) from the vector x.
Z = zeros(n,m,k);
for r = 1:k
    Z(:,:,r) = reshape(x(1:n*m),[n m]); x(1:n*m) = []; 
end
 
if opts.max_norm_regu.flag
idx_poiss = opts.task_type ==3;
if sum(idx_poiss)>0 
        
    Z_poiss = Z(:,idx_poiss,:);
      
    max_norm = max(abs(Z_poiss(:)));
    th  = opts.max_norm_regu.Z.param(1);
    if max_norm > th
        Z_poiss(abs(Z_poiss)>th) = Z_poiss(abs(Z_poiss)>th)/(2*max_norm);
    end
     
    Z(:,idx_poiss,:) = Z_poiss;
end
end

% soft_thresh for each parameter
if opts.sep_pen_inside.Z.flag && (~opts.sep_pen.flag)
    if ~strcmp(opts.prox_param.Z.method,'L1')
        for i_type = 1:3
            idx =  opts.task_type == i_type;
            if sum(idx)==0
                continue
            end
            param_tmp = sum(opts.prox_param.Z.param(i_type,:));
            Z(:,idx,:) = sub_prox_fun(Z(:,idx,:),t,opts.prox_param.Z.method,param_tmp,...
                opts.prox_param.Z.fix_feature_flag,opts.prox_param.Z.fix_feature(:,idx,:));
        end
    else

        for r = 1:k
            for i_type = 1:3
                idx =  opts.task_type == i_type;
                if sum(idx)==0
                    continue
                end
                Z(:,idx,r) = sub_prox_fun(Z(:,idx,r),t,opts.prox_param.Z.method,opts.prox_param.Z.param(i_type,r),...
                   opts.prox_param.Z.fix_feature_flag,opts.prox_param.Z.fix_feature(:,idx,r));
            end
        end
    end
else

if ~strcmp(opts.prox_param.Z.method,'L1')
   param_tmp = sum(opts.prox_param.Z.param);
   Z = sub_prox_fun(Z,t,opts.prox_param.Z.method,param_tmp,...
        opts.prox_param.Z.fix_feature_flag,opts.prox_param.Z.fix_feature);
else
    for r = 1:k
        Z(:,:,r) = sub_prox_fun(Z(:,:,r),t,opts.prox_param.Z.method,opts.prox_param.Z.param(r),...
            opts.prox_param.Z.fix_feature_flag,opts.prox_param.Z.fix_feature(:,:,r));
    end
end



end
 
 
 
% re-organize the parameters into the parameter vector x.
v = [Z(:);x];


end