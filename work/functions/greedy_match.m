function [order_Y,Y]= greedy_match(X,Y)

 
mat = pdist2(X,Y);
 
n = size(mat,1);

% mat = mat + inf * eye(n);

order_Y = zeros(n,1);
for i = 1:n
   
    min_val = min(mat(:));
    [row,col] = find(mat == min_val);
    order_Y(row(1)) = col(1);
    mat(row(1),:) = inf;
    mat(:,col(1)) = inf;
    
end

Y = Y(order_Y,:);




end