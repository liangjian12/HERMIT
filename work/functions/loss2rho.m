function rho = loss2rho(sum_L,pi,k)

% mean_L =  mean(sum_L,2);
% sum_L = bsxfun(@minus,sum_L,mean_L);
% rho = exp(sum_L);
% rho = bsxfun(@times,rho,pi);    
% rho = bsxfun(@rdivide,rho,sum(rho,2)); 


sum_L = bsxfun(@plus,sum_L,log(max(pi,eps)));
mean_L =  mean(sum_L,2);
sum_L = bsxfun(@minus,sum_L,mean_L);
rho = softmax(sum_L')';

if any(isnan(rho(:))) || any(isinf(rho(:)))
    rho(isnan(rho)) = 1/k;
    rho(isinf(rho)) = 1/k;
    rho = bsxfun(@rdivide,rho,sum(rho,2)); 
end
rho = max(rho,0);
rho = min(rho,1);

end