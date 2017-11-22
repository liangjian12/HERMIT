function mu = softmax_pred(X,W)

G = X*W ;

mean_G  =  mean(G ,2);
G  = bsxfun(@minus,G ,mean_G );

mu = softmax(G')';



end