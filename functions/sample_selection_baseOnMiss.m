function idx = sample_selection_baseOnMiss(Omega,idx1,idx2)

m1 = length(idx1);
m2 = length(idx2);
score1 = sum(Omega(:,idx1),2)/m1;
score2 = sum(Omega(:,idx2),2)/m2;

score = score1.*score2;
[~,idx] = sort(score,'descend');

end