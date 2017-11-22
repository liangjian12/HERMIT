function X = sub_prox_fun(X,t,method,param,fix_feature_flag,fix_feature)

if fix_feature_flag == false
 
    
if strcmp(method,'L1')
  X = sign(X) .* max(abs(X) -  t*param(1),0);
  
elseif strcmp(method,'L2')
  X = X./(1+t*param(1));
elseif strcmp(method,'EN')    
  X = sign(X) .* max(abs(X) - t*param(1),0)./(1+t*param(2));  
elseif strcmp(method,'GS')       
  if  size(X,2)<2
      error('To choose Group Sparsity as the type of proximal operator, the second size of X should be bigger than 1')
  end
  norm_L2_2 = max(eps,sum(X.*X,2).^0.5);
  coef = max(0,1 -  t* param(1)./norm_L2_2);
  X = bsxfun(@times,X,coef);
  
%   X = trace_projection(X, t* param(1));
  
elseif strcmp(method,'LR')     
    
  if  size(X,2)<2
      error('To choose Group Sparsity as the type of proximal operator, the second size of X should be bigger than 1')
  end
  X = trace_projection(X, t* param(1));
  
elseif strcmp(method,'GS3')       
  if  size(X,3)<2
      error('To choose Group Sparsity as the type of proximal operator, the third size of X should be bigger than 1')
  end
  norm_L2_2 = max(eps,sum(X.*X,3).^0.5);
  coef = max(0,1 -   t* param(1)./norm_L2_2);
  X = bsxfun(@times,X,coef);
elseif strcmp(method,'GS_row')       
    norm_L2_2 = max(eps,sum(sum(X.*X,3),2).^0.5);
    coef = max(0,1 -  t* param(1)./norm_L2_2);
    X = bsxfun(@times,X,coef);
elseif strcmp(method,'GS_col')       
    norm_L2_2 = max(eps,sum(sum(X.*X,3),1).^0.5);
    coef = max(0,1 -  t* param(1)./norm_L2_2);
    X = bsxfun(@times,X,coef);
end


 


else
    
     
   X = fix_feature.*X;
   
    
end

    
%     norm_2 = sum(X.*X,1).^0.5;
%     X = bsxfun(@rdivide,X,max(norm_2,0.01));
    

end