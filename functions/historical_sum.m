function [X] = historical_sum(X,w)

n = size(X,1); 
for i = 1:n-1
    X(i+1,:) = w(1) * X(i+1,:) + w(2) *X(i,:);
end

end