function mi = mi_prob(P,Q)

p = mean(P);%p = p/max(sum(p),eps);
q = mean(Q);%q = q/max(sum(q),eps);

PQ = P'*Q/size(P,1);
% PQ = PQ/max(sum(PQ(:)),eps);

pq = p'*q;


PQ = max(PQ,1e-40);
pq = max(pq,1e-40);

a = PQ.*log(PQ./pq);

    
mi =  max(0,sum(a(:)));


end