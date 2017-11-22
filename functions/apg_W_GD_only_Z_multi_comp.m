function [Z] = apg_W_GD_only_Z_multi_comp(X,Y,opts)
% Input:

%Output:
 

% store data
opts.X = X;
opts.Y = Y;

% super parameters
k = opts.k;         % number of columns of Fi, i = 1,2,3; 
[n,d] = size(X);
[n,m] = size(Y);


 
%dimension of all parameters
dim_x = n*m*k;

%initialization of all parameters
if  opts.warm_start.flag || opts.warmFlag %
    Z_tmp = opts.init.Z;
    opts_tmp.X_INIT = [Z_tmp(:);];
else
    opts_tmp.X_INIT = opts.initial_scale *  randn(dim_x,1);
end
%APG
 
[x] = apg(@grad_W_GD_only_Z_multi_comp, @soft_thresh_W_GD_only_Z_multi_comp_sep, dim_x, opts); % x is a vector, containing all the parameters    
 

% re-organize the parameters into the forms of (W,Z,Phi) from the vector x.
Z = zeros(n,m,k);
for r = 1:k
    Z(:,:,r) = reshape(x(1:n*m),[n m]); x(1:n*m) = [];
end

end