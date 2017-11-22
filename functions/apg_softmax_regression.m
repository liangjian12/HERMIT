function [W] = apg_softmax_regression(X,rho,opts)
% Input:

%Output:
 

% store data
opts.X = X;
opts.rho = rho;

% super parameters
[n,d] = size(X);
[n,m] = size(rho);


 
%dimension of all parameters
dim_x = d * m ;

%initialization of all parameters
opts.X_INIT = 1e-5 *  randn(dim_x,1)/m;


%APG
[x] = apg(@grad_softmax_regression, @soft_thresh_softmax_regression, dim_x, opts); % x is a vector, containing all the parameters


% re-organize the parameters into the forms of (W,Z,Phi) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 
 
 

end