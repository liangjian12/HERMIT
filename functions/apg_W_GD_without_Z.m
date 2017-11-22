function [W,Phi] = apg_W_GD_without_Z(X,Y,opts)
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
dim_x = d * m ;

%initialization of all parameters
if opts.warm_start.flag || opts.warmFlag %
opts.X_INIT = opts.init.W(:);
else
opts.X_INIT = opts.initial_scale *  randn(dim_x,1);
end

%APG

% [x,opts] = apg_plus(@grad_W_GD_without_Z, @soft_thresh_W_GD_without_Z, @opti_close_Phi_without_Z, dim_x, opts); % x is a vector, containing all the parameters
if opts.flag_decomp
[x] = apg_decomp(@grad_W_GD_without_Z, @soft_thresh_W_GD_without_Z_decomp, dim_x, opts); % x is a vector, containing all the parameters    
else
[x] = apg(@grad_W_GD_without_Z, @soft_thresh_W_GD_without_Z, dim_x, opts); % x is a vector, containing all the parameters
end

% re-organize the parameters into the forms of (W,Z,Phi) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Phi = []; 

% Phi = opts.Phi; 
 

end