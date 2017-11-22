function [W,Phi,pi] = solver_ADA_without_Z(X,Y,k,opts,prox_param)
 
prox_param.W.ada_lasso_flag = false; 
prox_param.W.ada_lasso_weight = {}; 
   
[W,Phi,pi] = solver_sub_without_Z(X,Y,k,opts,prox_param);


    
prox_param.W.ada_lasso_flag = true; 

if length(size(W)) == 2
    for r = 1:k
        prox_param.W.ada_lasso_weight{r} = 1./(abs(W(:,r))+eps);
    end
elseif length(size(W)) == 3
    for r = 1:k
        prox_param.W.ada_lasso_weight{r} = 1./(abs(W(:,:,r))+eps);
    end
end
    

[W,Phi,pi] = solver_sub_without_Z(X,Y,k,opts,prox_param);

end