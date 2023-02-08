function [newx, setting] = zscore(x,setting)

if nargin < 2
    setting.mean = mean(x);
    setting.std = std(x);
end

setting.std(setting.std==0) = inf;

nrow = size(x,1);

m = repmat(setting.mean,nrow,1);
s = repmat(setting.std,nrow,1);

newx = (x-m)./s;