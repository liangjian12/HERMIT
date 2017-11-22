function [a,K_new] = acq_fun(x_new,opts)

K_old = opts.K;
X_old = opts.X;
Y_old = opts.Y;

rho = opts.mcmc_rho;

D_new = pdist2(X_old,x_new);
D_new = D_new * rho;
k_new = kernel_bayes(D_new,'ard');
K_new = [K_old k_new;k_new' 1];

sigma_0 = opts.sigma_0;
n = size(K_old,1);
mu = k_new'*((K_old + sigma_0^2 * eye(n))\Y_old);
sigma = 1 + sigma_0^2 - k_new'*((K_old + sigma_0^2 * eye(n))\k_new);

kappa = opts.kappa;

a =  mu - kappa * sigma ;


end