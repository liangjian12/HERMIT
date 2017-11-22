function nmi = nmi_prob(P,Q)

I = mi_prob(P,Q);
Hp = mi_prob(P,P);
Hq = mi_prob(Q,Q);

if I < 1e-10
    I = 0;
end

nmi = I./max(max(Hp,Hq),1e-16);

if nmi > 1.05
    disp('')
end


end