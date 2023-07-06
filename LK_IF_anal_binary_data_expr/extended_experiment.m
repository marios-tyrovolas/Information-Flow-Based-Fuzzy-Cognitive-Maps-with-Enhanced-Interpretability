%%
clc
clear all
%% X.San Liang's Experiment Extesion
% Can I apply the IF-based causality analysis between a continuous and a discrete variable, or do I have to binarize the 
% continuous variable and then redid the IF analysis?

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1_original = x1;
x2_original = x2;
x3_original = x3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1)
clf
plot([x1 x2 x3])
axis([2500 2800 -4 4])
xlabel('Time steps')
print -depsc series.eps


 [T21, err90_21, err95_21, err99_21] = multi_causality_est([x1 x2 x3], 1);
 [T12, err90_12, err95_12, err99_12] = multi_causality_est([x2 x1 x3], 1);
 [T23, err90_23, err95_23, err99_23] = multi_causality_est([x3 x2 x1], 1);
 [T32, err90_32, err95_32, err99_32] = multi_causality_est([x2 x3 x1], 1);
 [T13, err90_13, err95_13, err99_13] = multi_causality_est([x3 x1 x2], 1);
 [T31, err90_31, err95_31, err99_31] = multi_causality_est([x1 x3 x2], 1);


TTT = ...
[0  T21 T31 
 T12 0 T32 
 T13 T23 0]; 

err = ...
[0  err99_21 err99_31 
 err99_12 0 err99_32 
 err99_13 err99_23 0];
 

fprintf('T = \n');
fprintf('[  \\   T2->1  T3->1\n');
fprintf(' T1->2   \\    T3->2\n');
fprintf(' T1->3  T2->3     \\] = \n');
fprintf('                       %5.2f  %5.2f  %5.2f \n', TTT') 
	% To match the above TTT, the printed version shoould be TTT'.
	% That is to say, shown here is actually the real TTT 
fprintf('err99 =\n');
fprintf('%5.2f  %5.2f  %5.2f \n', err') 


%fid = fopen('T_noise.dat', 'w');
%fprintf(fid, '%6.3f  %6.3f  %6.3f \n', TTT');
%	% this saved T.dat is actually the matrix TTT
%fclose(fid);
%
%fid1 = fopen('err_noise.dat', 'w');
%fprintf(fid1, '%6.3f  %6.3f  %6.3f\n', err');
%fclose(fid1);
%

%T's & Confidence Level for each case
p = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IFs = cell(8,1);
ERR = cell(8,1);
IFs{p} = TTT;
ERR{p} = err';
p = p+1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% If binarized
 for n=1:nm,
     if x1(n,1)>0, x1(n,1)=1; else, x1(n,1)=0; end
     if x2(n,1)>0, x2(n,1)=1; else, x2(n,1)=0; end
     if x3(n,1)>0, x3(n,1)=1; else, x3(n,1)=0; end
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1_binarized = x1;
x2_binarized = x2;
x3_binarized = x3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2)
clf
plot([x1 x2 x3])
axis([2000 2100 0 1])
xlabel('Time steps')
title('Binarized series')
print -depsc series_binarized.eps


 [T21, err90_21, err95_21, err99_21] = multi_causality_est([x1 x2 x3], 1);
 [T12, err90_12, err95_12, err99_12] = multi_causality_est([x2 x1 x3], 1);
 [T23, err90_23, err95_23, err99_23] = multi_causality_est([x3 x2 x1], 1);
 [T32, err90_32, err95_32, err99_32] = multi_causality_est([x2 x3 x1], 1);
 [T13, err90_13, err95_13, err99_13] = multi_causality_est([x3 x1 x2], 1);
 [T31, err90_31, err95_31, err99_31] = multi_causality_est([x1 x3 x2], 1);


TTT = ...
[0  T21 T31 
 T12 0 T32 
 T13 T23 0]; 

err = ...
[0  err99_21 err99_31 
 err99_12 0 err99_32 
 err99_13 err99_23 0];


fprintf('\n\n\n If the series are binarized, then\n');
fprintf('T_binary = \n');
fprintf('[  \\   T2->1  T3->1\n');
fprintf(' T1->2   \\    T3->2\n');
fprintf(' T1->3  T2->3     \\] = \n');
fprintf('                       %5.2f  %5.2f  %5.2f \n', TTT') 
	% To match the above TTT, the printed version shoould be TTT'.
	% That is to say, shown here is actually the real TTT 
fprintf('err99_binary =\n');
fprintf('%5.2f  %5.2f  %5.2f \n', err') 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IFs{p} = TTT;
ERR{p} = err';
p = p+1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%fid = fopen('T_noise.dat', 'w');
%fprintf(fid, '%6.3f  %6.3f  %6.3f \n', TTT');
%	% this saved T.dat is actually the matrix TTT
%fclose(fid);
 
%fid1 = fopen('err_noise.dat', 'w');
%fprintf(fid1, '%6.3f  %6.3f  %6.3f\n', err');
%fclose(fid1);

%% Extension of the Experiment
% Mixed Cases 

%All possible combinations
combinations = cell(6,1);
combinations{1} = [x1_binarized,x2_original,x3_original];
combinations{2} = [x1_binarized,x2_binarized,x3_original];
combinations{3} = [x1_original,x2_binarized,x3_original];
combinations{4} = [x1_original,x2_binarized,x3_binarized];
combinations{5} = [x1_original,x2_original,x3_binarized];
combinations{6} = [x1_binarized,x2_original,x3_binarized];

for i=1:height(combinations)
    

    figure
    clf
    subplot(3,1,1)
    plot(combinations{i}(:,1))
    axis([2000 2100 -4 4])
    xlabel('Time steps')
    title('X1')
    subplot(3,1,2)
    plot(combinations{i}(:,2))
    axis([2000 2100 -4 4])
    xlabel('Time steps')
    title('X2')
    subplot(3,1,3)
    plot(combinations{i}(:,3))
    axis([2000 2100 -4 4])
    xlabel('Time steps')
    title('X3')
    s = strcat('Mixed case ',num2str(i));
    sgtitle(s) 
    
     [T21, err90_21, err95_21, err99_21] = multi_causality_est([combinations{i}(:,1) combinations{i}(:,2) combinations{i}(:,3)], 1);
     [T12, err90_12, err95_12, err99_12] = multi_causality_est([combinations{i}(:,2) combinations{i}(:,1) combinations{i}(:,3)], 1);
     [T23, err90_23, err95_23, err99_23] = multi_causality_est([combinations{i}(:,3) combinations{i}(:,2) combinations{i}(:,1)], 1);
     [T32, err90_32, err95_32, err99_32] = multi_causality_est([combinations{i}(:,2) combinations{i}(:,3) combinations{i}(:,1)], 1);
     [T13, err90_13, err95_13, err99_13] = multi_causality_est([combinations{i}(:,3) combinations{i}(:,1) combinations{i}(:,2)], 1);
     [T31, err90_31, err95_31, err99_31] = multi_causality_est([combinations{i}(:,1) combinations{i}(:,3) combinations{i}(:,2)], 1);


    TTT = ...
    [0  T21 T31 
     T12 0 T32 
     T13 T23 0]; 

    err = ...
    [0  err99_21 err99_31 
     err99_12 0 err99_32 
     err99_13 err99_23 0];


    
    fprintf('T_binary = \n');
    fprintf('[  \\   T2->1  T3->1\n');
    fprintf(' T1->2   \\    T3->2\n');
    fprintf(' T1->3  T2->3     \\] = \n');
    fprintf('                       %5.2f  %5.2f  %5.2f \n', TTT') 
        % To match the above TTT, the printed version shoould be TTT'.
        % That is to say, shown here is actually the real TTT 
    fprintf('err99_mixed =\n');
    fprintf('%5.2f  %5.2f  %5.2f \n', err') 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    IFs{p} = TTT;
    ERR{p} = err';
    p = p+1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
