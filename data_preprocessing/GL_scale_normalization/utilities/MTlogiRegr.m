function theta = MTlogiRegr(trainData, trainClass, w)

if nargin < 3
    u = 0;
    sigmaInv = 0;
else    
    if ~isinf(1./var(w,0,2))
        u = mean(w,2);
        sigmaInv = diag(1./var(w,0,2));
    else
        u = 0;
        sigmaInv = 0;
    end
end

[nEx, nFea] = size(trainData);

trainData = [ones(nEx,1) trainData];

nFea = nFea + 1;

yi = repmat(trainClass,1,size(trainData,2));

x = yi.*trainData;

initialTheta = ones(nFea,1)/nFea;

f = @(theta) objF(theta,x,u,sigmaInv);
fp = @(theta) DobjF(theta,x,u,sigmaInv);
fpp = @(theta) DDobjF(theta,x,u,sigmaInv);

ff = {f;fp;fpp};

% optTheta = CG_NR(initialTheta,f,fp,fpp);

options = optimoptions('fminunc',...
    'GradObj','on',...
    'Algorithm','trust-region',...
    'Hessian','on',...
    'Display','off',...
    'MaxIter',30);
optTheta = fminunc(ff,initialTheta,options);
% options = optimset('maxIter',1000);
% options = optimset('GradObj','on','maxIter',1000,'Display','iter');
% initialTheta = rand(nFea,1)*1e-2;
% 
% J = @(theta) costFunction(theta,trainData,trainClass,u,sigmaInv);
% 
% [optTheta, ~, ~] ...
%     = fminunc(J, initialTheta,options);

% result{1} = functionVal;
% result{2} = exitFlag;

theta = optTheta;

function out = objF(theta,x, u,sigmaInv)

nEx = size(x,1);
out = 0;

for iEx = 1:nEx
    
    out = out + log(1./(sigm(theta,x(iEx,:))));
    
end

out = 1/nEx*out + 1/2*(theta-u)'*sigmaInv*(theta-u);

function out = DobjF(theta,x, u,sigmaInv)

nEx = size(x,1);
out = 0;

for iEx = 1:nEx
    out = out + (sigm(theta,x(iEx,:))-1)*x(iEx,:);
end

out = 1/nEx * out' + sigmaInv*(theta-u);


function out = DDobjF(theta,x, u,sigmaInv)
nEx = size(x,1);
out = 0;

for iEx = 1:nEx
    out = out + (1 - sigm(theta,x(iEx,:))) * x(iEx,:)'*x(iEx,:);
end

out = 1/nEx * out + sigmaInv;


% function [jVal, jGrad] = costFunction (theta,trainData,trainClass, u,sigmaInv)
% nEx = length(trainClass);
% Htheta = sigm(theta,trainData);
% 
% jVal = -1/nEx*(1*trainClass'*log(Htheta) + ...
%     (1-trainClass)'*log(1-Htheta+eps))+1/2*(theta-u)'*sigmaInv*(theta-u);
% 
% jGrad = 1/nEx * ((trainData'*(Htheta-trainClass)))+sigmaInv*(theta-u);


function out = sigm (theta,x)

out = 1./(1+exp(-theta'*x'))'*(1-eps);

if out == 0 
    out = eps;
end