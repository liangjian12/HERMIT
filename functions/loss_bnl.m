function l = loss_bnl(y,g)
    b = g + log(1+exp(-g));
    flag = isinf(b);
    if any(flag(:))
        b = g + max(0,1-g)-1;
    end
    l = (y.*g - b); 
end