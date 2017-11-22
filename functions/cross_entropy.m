function en = cross_entropy(P,Q)

en = -sum(P.*log(max(Q,eps)),2);

end