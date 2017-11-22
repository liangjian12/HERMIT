function [W,Z,Phi] = apg_W_GD(X,Y,opts)
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
dim_x = d * m + n*m;

%initialization of all parameters
opts.X_INIT = opts.initial_scale *  randn(dim_x,1);

%APG
[x,opts] = apg_plus(@grad_W_GD, @soft_thresh_W_GD, @opti_close_Phi, dim_x, opts); % x is a vector, containing all the parameters

% re-organize the parameters into the forms of (W,Z,Phi) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
Z = reshape(x(1:n*m),[n m]); x(1:n*m) = [];
Phi = opts.Phi; 
 

end