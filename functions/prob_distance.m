function d = prob_distance(P,Q,opt)

if strcmp(opt,'KL')
    
    Q = max(Q,eps);
    P = max(P,eps);
    
    d = mean(sum(P.*log(P./Q),2));
    
    
elseif strcmp(opt,'sKL')
    
    Q = max(Q,eps);
    P = max(P,eps);
    
    d = mean(sum(P.*log(P./Q),2)) + mean(sum(Q.*log(Q./P),2));
    d = d/2;
    
elseif strcmp(opt,'JS')
    
    
    Q = max(Q,eps);
    P = max(P,eps);
    M = (P+Q)/2;
    
    d = mean(sum(P.*log(P./M),2)) + mean(sum(Q.*log(Q./M),2));
    
    d = d/2;
    
elseif strcmp(opt,'Hellinger')
    
    P = sqrt(P);
    Q = sqrt(Q);
    
    d = norm(P(:)-Q(:));
    
    
elseif strcmp(opt,'ABS')
    
    d = mean(sum( abs(P-Q),2));
    
elseif strcmp(opt,'crossentropy')
    
    d = -mean(sum(P.*log(max(Q,eps)),2));
    
elseif strcmp(opt,'nmi_prob')
    
    d = -nmi_prob(P,Q);
    
end


end