function dx = grad_softmax_regression(x, opts)

 
X = opts.X;
rho = opts.rho;

[n,d] = size(X);
[n,m] = size(rho);
 

 
% re-organize the parameters into the forms of (b,w,F,C) from the vector x.
W = reshape(x(1:d*m),[d m]); x(1:d*m) = []; 

G = X*W ;

mean_G  =  mean(G ,2);
G  = bsxfun(@minus,G ,mean_G );

mu = softmax(G')';
err = rho - mu;

num = sum(rho);
err = bsxfun(@rdivide,err,num) * n / m;

dW = X'*err;
dW = dW/size(X,1);
 

dx = [dW(:)];
dx =  - dx; 



end