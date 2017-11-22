function output = nmi(label_cluster,label_true)
    
    K = max(label_true);
    N = length(label_true);
    N_cluster = zeros(K,1);
    N_true = zeros(K,1);
    N_each = zeros(K,K);
    
    for i = 1:K
        N_cluster(i) = sum(label_cluster == i);
        N_true(i) = sum(label_true == i);
    end
    
    for i = 1:K
       ind_i = find(label_cluster == i);
       for j = 1:K          
          ind_j = find(label_true == j);
          ind = intersect(ind_i, ind_j);
          N_each(i,j) = length(ind);
       end
    end
    
    %num = 0;
    %for i = 1:K      
    %   for j = 1:K          
    %      if N_each(i,j)
    %        num = num + N_each(i,j) * log((N * N_each(i,j))/(N_cluster(i)*N_true(j)));
    %      end
    %   end
    %end
    
    N_each_revise = N_each;
    N_each_revise(N_each_revise==0)=1;
    num = sum(sum(N_each.*log((N*N_each_revise)./(N_cluster*N_true'))));
   
    den = sqrt( sum( N_cluster .* log( N_cluster / N ) ) * sum( N_true .* log( N_true / N ) ) );
    
    output = num/den;

end