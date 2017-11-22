function [a,K_new] = acq_fun_m(X_new,opts)

K_old = opts.K;
X_old = opts.X;
Y_old = opts.Y;

rho = opts.mcmc_rho;

D_old_new = pdist2(X_old,X_new);
D_old_new = D_old_new * rho;
K_old_new = kernel_bayes(D_old_new,'ard');

D_new_new = pdist(X_new);
D_new_new = squareform(D_new_new);
D_new_new = D_new_new * rho;
K_new_new = kernel_bayes(D_new_new,'ard');

K_new = [K_old K_old_new;K_old_new' K_new_new];

sigma_0 = opts.sigma_0;
n_old = size(K_old,1);
n_new = size(K_new_new,1);
mu = K_old_new'*((K_old + sigma_0^2 * eye(n_old))\Y_old);
sigma = K_new_new + sigma_0^2 * eye(n_new) - K_old_new'*((K_old + sigma_0^2 * eye(n_old))\K_old_new);

kappa = opts.kappa;

a =  mu - kappa * diag(sigma) ;


end