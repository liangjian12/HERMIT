function Y = link_fun_rnd(mu,Phi,options)

[n,m] = size(mu);
% Phi = Phi;

if strcmp(options,'gauss')
    for i = 1:n
        for j = 1:m            
            Y(i,j) = normrnd(mu(i,j),1/Phi(j,j));            
        end
    end
elseif strcmp(options,'bnl')
    for i = 1:n
        for j = 1:m            
            Y(i,j) = binornd(1,mu(i,j));            
        end
    end    
elseif strcmp(options,'poiss')    
    for i = 1:n
        for j = 1:m            
            Y(i,j) = poissrnd(mu(i,j));            
        end
    end
end



end