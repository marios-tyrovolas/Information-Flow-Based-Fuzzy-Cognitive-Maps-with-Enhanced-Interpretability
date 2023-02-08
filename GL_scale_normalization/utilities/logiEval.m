function pred = logiEval(theta, testData)

testData = [ones(size(testData,1),1) testData];

pred = sigm(theta,testData);


function out = sigm (theta,x)

out = 1./(1+exp(-theta'*x'))';