%%
clc;
clear all;
%%
% Generate a VAR(3) process with the following coefficients
%
% A0 = [a10, a20, a30]';
%
% A = (a_ij) = [a11   a12   a13
%	        a21   a22   a23
%		a31   a32   a33];
%
% B = [b1 b2 b3]';
%
%
% x3 as a confounder
%             x3
%       x1           x2
%
% x3-->x1,   x3-->x2,   no other causality exists
%

  
  A = [ 0.4   0   -0.8 
        0    -0.8  0.7
        0     0    0.5];

  B  = [1 0 0
	0 1 0
	0 0 1];

  A0 = [0.1  0.7  0.5]';
% A0 = [0  0  0]';

nm = 10000;


Xn = randn(3,1);	% random initialization

for n = 1 : nm,
 	x1(n,1) = Xn(1);
 	x2(n,1) = Xn(2);
	x3(n,1) = Xn(3);
    e = randn(3,1);
    Xn1 = A0 + A * Xn + B * e;
    Xn = Xn1;
end



figure(1)
clf
plot([x1 x2 x3])
axis([2500 2800 -4 4])
xlabel('Time steps')
%print -depsc series.eps


 [T21, err90_21_normal, err95_21_normal, err99_21_normal] = multi_causality_est([x1 x2 x3], 1);
 [T12, err90_12_normal, err95_12_normal, err99_12_normal] = multi_causality_est([x2 x1 x3], 1);
 [T23, err90_23_normal, err95_23_normal, err99_23_normal] = multi_causality_est([x3 x2 x1], 1);
 [T32, err90_32_normal, err95_32_normal, err99_32_normal] = multi_causality_est([x2 x3 x1], 1);
 [T13, err90_13_normal, err95_13_normal, err99_13_normal] = multi_causality_est([x3 x1 x2], 1);
 [T31, err90_31_normal, err95_31_normal, err99_31_normal] = multi_causality_est([x1 x3 x2], 1);


TTT_normal = ...
[0  T21 T31 
 T12 0 T32 
 T13 T23 0]; 

err_normal = ...
[0  err99_21_normal err99_31_normal 
 err99_12_normal 0 err99_32_normal 
err99_13_normal err99_23_normal 0];
 

fprintf('T = \n');
fprintf('[  \\   T2->1  T3->1\n');
fprintf(' T1->2   \\    T3->2\n');
fprintf(' T1->3  T2->3     \\] = \n');
fprintf('                       %5.4f  %5.4f  %5.4f \n', TTT_normal') 
	% To match the above TTT, the printed version shoould be TTT'.
	% That is to say, shown here is actually the real TTT 
fprintf('err99 =\n');
fprintf('%5.4f  %5.4f  %5.4f \n', err_normal') 


%fid = fopen('T_noise.dat', 'w');
%fprintf(fid, '%6.3f  %6.3f  %6.3f \n', TTT');
%	% this saved T.dat is actually the matrix TTT
%fclose(fid);
%
%fid1 = fopen('err_noise.dat', 'w');
%fprintf(fid1, '%6.3f  %6.3f  %6.3f\n', err');
%fclose(fid1);
%



% If binarized
 for n=1:nm,
     if x1(n,1)>0, x1(n,1)=1; else, x1(n,1)=0; end
     if x2(n,1)>0, x2(n,1)=1; else, x2(n,1)=0; end
     if x3(n,1)>0, x3(n,1)=1; else, x3(n,1)=0; end
 end



figure(2)
clf
plot([x1 x2 x3])
axis([2000 2100 0 1])
xlabel('Time steps')
title('Bindarized series')
%print -depsc series_binarized.eps


 [T21, err90_21_binary, err95_21_binary, err99_21_binary] = multi_causality_est([x1 x2 x3], 1);
 [T12, err90_12_binary, err95_12_binary, err99_12_binary] = multi_causality_est([x2 x1 x3], 1);
 [T23, err90_23_binary, err95_23_binary, err99_23_binary] = multi_causality_est([x3 x2 x1], 1);
 [T32, err90_32_binary, err95_32_binary, err99_32_binary] = multi_causality_est([x2 x3 x1], 1);
 [T13, err90_13_binary, err95_13_binary, err99_13_binary] = multi_causality_est([x3 x1 x2], 1);
 [T31, err90_31_binary, err95_31_binary, err99_31_binary] = multi_causality_est([x1 x3 x2], 1);


TTT_binary = ...
[0  T21 T31 
 T12 0 T32 
 T13 T23 0]; 

err_binary = ...
[0  err99_21_binary err99_31_binary 
 err99_12_binary 0 err99_32_binary 
err99_13_binary err99_23_binary 0];


fprintf('\n\n\n If the series are binarized, then\n');
fprintf('T_binary = \n');
fprintf('[  \\   T2->1  T3->1\n');
fprintf(' T1->2   \\    T3->2\n');
fprintf(' T1->3  T2->3     \\] = \n');
fprintf('                       %5.4f  %5.4f  %5.4f \n', TTT_binary') 
	% To match the above TTT, the printed version shoould be TTT'.
	% That is to say, shown here is actually the real TTT 
fprintf('err99_binary =\n');
fprintf('%5.4f  %5.4f  %5.4f \n', err_binary') 


%fid = fopen('T_noise.dat', 'w');
%fprintf(fid, '%6.3f  %6.3f  %6.3f \n', TTT');
%	% this saved T.dat is actually the matrix TTT
%fclose(fid);
 
%fid1 = fopen('err_noise.dat', 'w');
%fprintf(fid1, '%6.3f  %6.3f  %6.3f\n', err');
%fclose(fid1);


