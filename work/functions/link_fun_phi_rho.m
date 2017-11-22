function [phi] = link_fun_phi_rho(y,g,Omega,rho,options)

[n,m] = size(y);
  n = sum(rho);
if strcmp(options,'gauss')
    
    yg = sum(y.*g,1);
    y2 = sum(y.*y,1);

%     n = sum(rho);
    
%     phi = (2*yg+sqrt(4*yg.^2 + 8 * y2*n))./(4*y2);
    phi = (yg+sqrt(yg.^2 + 4 * y2*n))./(2*y2 + eps);
    
    phi = min(phi,2);
    phi = max(phi,0.5);
    
    phi = diag(phi);
    
%      n = sum(rho);
     err = (y-g).*Omega;
%      err_rho = bsxfun(@times,err,rho);
     phi = diag(1./(diag(err'*err/(n-1))).^0.5);
     
     disp('')
     
%      phi = 2 * eye(m);
    
elseif strcmp(options,'bnl')
    
    phi = eye(m);
    
elseif strcmp(options,'poiss')      
    phi = eye(m); 
end

end
