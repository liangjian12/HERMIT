function v = soft_thresh_W_GD_without_Z_decomp(x, t, opts, decomp_id)

X = opts.X;
Y = opts.Y;

[n,d] = size(X);
[n,m] = size(Y);

% re-organize the parameters into the forms of (W,Z) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 

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
end

% idx_bnl = opts.task_type ==2;
% if sum(idx_bnl)>0 
%     W_bnl = W(:,idx_bnl);
%     
%     W_bnl_1 = W_bnl(1,:);
%     W_bnl_rest = W_bnl(2:end,:);
%     
%     max_norm = max(abs(W_bnl_1(:)));
%     th  = opts.max_norm_regu.W.param(1);
%     if max_norm > th
%         W_bnl_1(abs(W_bnl_1)>th) = W_bnl_1(abs(W_bnl_1)>th)/(2*max_norm);
%     end
%     
%     max_norm = max(abs(W_bnl_rest(:)));
%     th  = opts.max_norm_regu.W.param(2);
%     if max_norm > th
%         W_bnl_rest(abs(W_bnl_rest)>th) = W_bnl_rest(abs(W_bnl_rest)>th)/(2*max_norm);
%     end
%     
%     W_bnl(1,:) = W_bnl_1;
%     W_bnl(2:end,:) = W_bnl_rest;
%  
%     W(:,idx_bnl) = W_bnl;
% end
end
 

% soft_thresh for each parameter
 
W(2:end,:) = sub_prox_fun(W(2:end,:),t,opts.prox_param.W.method_decomp{decomp_id},opts.prox_param.W.param_decomp{decomp_id},...
    opts.prox_param.W.fix_feature_flag,opts.prox_param.W.fix_feature(2:end,:));
 
 
 
% re-organize the parameters into the parameter vector x.
v = [W(:);x];

% norm_2 = sum(v.*v).^0.5;
% v = v/norm_2;


end