function [W,Phi] = apg_W_gaussClose_without_Z(X,Y,W,opts)
% Input:

%Output:
 

% store data
opts.X = X;
opts.Y = Y;

% super parameters
k = opts.k;         % number of columns of Fi, i = 1,2,3; 
[n,d] = size(X);
[n,m] = size(Y);
Phi = eye(m);
m = m - opts.task_num_each_type(1);

W_non_gauss = [];
W_gauss = [];

if m > 0
 
%dimension of all parameters
dim_x = d * m ;

%initialization of all parameters
opts.X_INIT = opts.initial_scale *  rand(dim_x,1);

opts_tmp = opts;
idx = opts.task_type > 1;
opts_tmp.Y = Y(:,idx);
opts_tmp.Omega = opts_tmp.Omega(:,idx);
opts_tmp.task_type = opts.task_type(idx);
opts_tmp.task_num_each_type(1) = 0;

%APG

[x] = apg_adadelta(@grad_W_gaussClose_without_Z, @soft_thresh_W_gaussClose_without_Z_adadelta, ...
     dim_x, opts_tmp); % x is a vector, containing all the parameters
  


% re-organize the parameters into the forms of (W,Z,Phi) from the vector x.
W_non_gauss = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 

end


if opts.task_num_each_type(1) > 0

 
    if strcmp(opts.gaussClose.method,'prime')
        idx = opts.task_type == 1;
        [W_gauss,Phi_gauss] = opti_close_gauss_prime(W(:,idx),opts);
        
    elseif strcmp(opts.gaussClose.method,'dual')
    
        idx = opts.task_type == 1;
        [W_gauss,Phi_gauss] = opti_close_gauss_dual(W(:,idx),opts);
        
    end



Phi(idx,idx) = Phi_gauss;


end
    
    
 
W = [W_gauss W_non_gauss];
    

end