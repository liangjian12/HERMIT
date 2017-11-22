function [phi] = link_fun_phi_rho_fix(y,g,Omega,rho,options)

[n,m] = size(y);
  
if strcmp(options,'gauss')
    
    yg = sum(y.*g,1);
    y2 = sum(y.*y,1);

%     n = sum(rho);
    
%     phi = (2*yg+sqrt(4*yg.^2 + 8 * y2*n))./(4*y2);
    phi = (yg+sqrt(yg.^2 + 4 * y2*n))./(2*y2);
    
    phi = diag(phi);
    
%      err = (y-g).*Omega;
%      err_rho = bsxfun(@times,err,rho);
%      phi = diag(diag(err'*err_rho/(n-1)));
     
     phi = 1 * eye(m);
    
elseif strcmp(options,'bnl')
    
    phi = eye(m);
    
elseif strcmp(options,'poiss')      
    phi = eye(m); 
end

end
