function c = generalLogiFit(x,y)

load('logisticFunctionCalculus')

M = interp1(y,x,1/2);
% options = optimset('PlotFcns',{@optimplotx,@optimplotfval});
fun = @(Q)1./(1+Q.*exp((log((1+Q).^3.3219-1)-log(Q))./(x(1)-M).*(x(end)-M)))-0.9.^(log(1+Q)/log(2))-0.01;
try
    Q = fzero(fun,eps);
catch
    Q = 1;
end
B = (log((1+Q)^3.3219-1)-log(Q))/(M-x(1));
v = log(1+Q)/log(2);

% B = log(9)/(x(end)-M);
% Q = 1;
% v = 1;
% if ~isinf(exp(-B*(x(1)-M)))
%     v = log10(1+exp(-B*(x(1)-M)));
% else
%     v = -B*(x(1)-M)/log(10);
% end

c = [B,M,Q,v]; % B M Q v
objF = @(c) logiCal(c,x,y);


% options = optimoptions('fminunc',...
%     'GradObj','on',...
%     'Algorithm','trust-region',...
%     'Hessian','on',...
%     'Display','off',...
%     'MaxIter',100);
% c = fminunc(objF,c,options);

options = optimoptions('fmincon',...
    'GradObj','on',...
    'Algorithm','trust-region-reflective',...
    'Hessian','user-supplied',...
    'Display','off',...
    'MaxIter',100);
% x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
lb = [eps;x(1);eps;eps];
ub = [inf;x(end);inf;inf];
c = fmincon(objF,c,[],[],[],[],lb,ub,[],options);


function [l,lp,lpp] = logiCal(c,x,y)

load('logisticFunctionCalculus')

n = length(x);
l = 0;
lp = 0;
lpp = 0;

for i = 1:n
    l = l + f(c(1),c(2),c(3),c(4),x(i),y(i));
    lp = lp + fp(c(1),c(2),c(3),c(4),x(i),y(i));
    lpp = lpp + fpp(c(1),c(2),c(3),c(4),x(i),y(i));
end

l = 1/n * l;
lp = 1/n * lp;
lpp = 1/n * lpp;

if isnan(sum(l)+sum(lp)+sum(sum(lpp))) || isinf(sum(l)+sum(lp)+sum(sum(lpp)))
    l = zeros(size(l));
    lp = zeros(size(lp));
    lpp = zeros(size(lpp));
end