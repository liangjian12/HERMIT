function [phi] = link_fun_phi(y,g,options)

%phi is a vector
[n,m] = size(y);
 
   
if strcmp(options,'gauss')
    
     err = y-g;
     phi = diag(diag(err'*err/n));
  
    
elseif strcmp(options,'bnl')
    
    phi = eye(m);
    
elseif strcmp(options,'poiss')    
    
    phi = eye(m);
    
end

end
