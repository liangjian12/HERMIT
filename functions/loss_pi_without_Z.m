function L = loss_pi_without_Z(rho,pi,gamma,lambda,L1_W)

n = size(rho,1);
H = bsxfun(@times,rho,log(pi+eps));

pi_gamma = pi.^gamma;

L = - sum(H(:))/n + lambda * sum(pi_gamma.*L1_W) ;

end