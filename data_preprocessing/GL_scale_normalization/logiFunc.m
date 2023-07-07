function out = logiFunc(c,x)

B = c(1);
M = c(2);
Q = c(3);
v = c(4);

out = 1./(1+Q*exp(-B*(x-M))).^(1/v);