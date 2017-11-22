function c = lambda_fun(n,p)

Mn = sqrt(log(n));
c = Mn*log(n)*sqrt(log(max(n,p))/n);

end