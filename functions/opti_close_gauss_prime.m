function  [W,Phi] = opti_close_gauss_prime(W,opts)

%  data

X = opts.X;
Y = opts.Y;

% super parameters
k = opts.k;         % number of columns of Fi, i = 1,2,3; 
[n,d] = size(X);
 
m =  opts.task_num_each_type(1);
rho = opts.rho;
Omega = opts.Omega;
 
 

idx = opts.task_type == 1;
Y = Y(:,idx);
Omega = Omega(:,idx);
X = bsxfun(@times,X,sqrt(rho));
Y = bsxfun(@times,Y,sqrt(rho));
iter_num = opts.gaussClose.maxIter;
 
n_vis = sum(Omega,1);

th =   bsxfun(@times,n_vis,ones(size(W))); 


 
for it = 1:iter_num
    
    G=X*W;
    
    [Phi] = link_fun_phi_rho(Y,G,Omega,rho,'gauss');
   
    S = zeros(d,m);
   
    W = X\(Y*Phi);
     
    W(2:end,:) = sub_prox_fun(W(2:end,:),th(2:end,:),opts.prox_param.W.method,opts.prox_param.W.param,...
    opts.prox_param.W.ada_lasso_flag,opts.prox_param.W.ada_lasso_weight(2:end,idx),...
    opts.prox_param.W.fix_feature_flag,opts.prox_param.W.fix_feature(2:end,idx));
    
    
end
 
 


end