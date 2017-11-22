function L = loss_pi(rho,pi,gamma,lambda,tau,L1_W,L1_Z)

n = size(rho,1);
H = bsxfun(@times,rho,log(pi+eps));

pi_gamma = pi.^gamma;

L = - sum(H(:))/n + lambda * sum(pi_gamma.*L1_W) + tau * sum(pi_gamma.*L1_Z);

end