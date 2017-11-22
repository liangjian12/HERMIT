function [l] = link_fun(y,g,phi,Omega,options)

%phi is a diagonal matrix
m = size(y,2);

if strcmp(options,'gauss')
    
    a =  phi ;
    y_phi = y*phi;
    b = g.^2/2;
%     c = -y.^2/(2*a) - log(2*pi)/2 - log(det(a))/2;
    c = bsxfun(@plus,-y_phi.^2/2,log(diag(a)')) - log(2*pi)/2;
    l = (y_phi.*g - b) + c;
    
    
elseif strcmp(options,'bnl')
    
    a = eye(m);
    b = g + log(1+exp(-g));
    flag = isinf(b);
    if any(flag(:))
        b = g + max(0,1-g)-1;
    end
    c = zeros(size(y));
    l = (y.*g - b);
    
    
elseif strcmp(options,'poiss')    
    
    a = eye(m);
    b = exp(g);
  
    c = zeros(size(y));
  
%     log_phi = diag(log(diag(phi)+eps));
   

%     l = (y.*g - b*phi+y*log_phi);
   l = (y.*g - b) - gammaln(y+1) ;
   
 
 
end
 



end