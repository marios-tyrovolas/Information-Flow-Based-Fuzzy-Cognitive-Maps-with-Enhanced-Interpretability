function ind = gencvind(allLb,nCV)

nLb = length(allLb);
allInd = 1:nLb;
ind = zeros(nLb,1);

posInd = allInd(allLb>0);
np = length(posInd);
negInd = allInd(allLb<=0);
nn = length(negInd);

ind(posInd) = crossvalind('Kfold',np,nCV);
ind(negInd) = crossvalind('Kfold',nn,nCV);
