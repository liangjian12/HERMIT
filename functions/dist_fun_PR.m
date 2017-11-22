function D = dist_fun_PR(w_true,w_out)

a =  bsxfun(@times,w_true,w_out);
D =  - sum(a,2)./(max(sum(w_out,2),eps));

 
end