function [accuracy_proportion] = accuracy(idx)
%CORRECTNESS Obtain proportion of data points correctly clustered together

correct = 0;
for i = 1:210

    if i<71
       correct = correct + double(idx(i)==mode(idx(1:70)));

    elseif i<141
       correct = correct + double(idx(i)==mode(idx(71:140)));

    else
        correct = correct + double(idx(i)==mode(idx(141:210)));
    end

end

accuracy_proportion = correct/210;
end

