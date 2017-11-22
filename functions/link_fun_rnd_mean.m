function Y = link_fun_rnd_mean(mu,Phi,options)

[n,m] = size(mu);
% Phi = Phi.^0.5;

if strcmp(options,'gauss')
    
    for i = 1:n
        for j = 1:m            
            Y(i,j) =   ( mu(i,j) +    randn)/Phi(j,j)  ;            
        end
    end
elseif strcmp(options,'bnl')
    Y = double(mu > 0.5);
elseif strcmp(options,'poiss')    
    Y = round(mu);
end



end