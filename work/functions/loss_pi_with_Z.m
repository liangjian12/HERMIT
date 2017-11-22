function L = loss_pi_with_Z(rho,pi,gamma,lambda,L1_W,tau,L1_Z)

n = size(rho,1);
H = bsxfun(@times,rho,log(pi+eps));

pi_gamma = pi.^gamma;

L = - sum(H(:))/n + lambda * sum(pi_gamma.*L1_W) +   sum(tau.*pi_gamma.*L1_Z)  ;

end