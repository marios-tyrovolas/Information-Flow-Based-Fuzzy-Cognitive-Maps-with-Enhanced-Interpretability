%%
clc
clear all
warning('off')
%% Particle Swarm Optimization

CostFunction = @(x) E(x); %cost function

%Number of variables
filepath_for_T_matrix = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\taus.csv");
taus = readtable(filepath_for_T_matrix);
taus = table2array(taus);
N = nnz(taus); %number of non-zero taus
nvars = N+1; %N+phi

% Option1: Causal relations (based on computed IFs) are imposed as constraints in the optimization problem (finding the weights). 
lb = [0;-1;-1;-1;-1;-1;-1;-1;-1;-1];
ub = [0.8;1;1;1;1;1;1;1;1;1];

% Option2: Take the normalized IFs as the weights but are subject to modification according to the confidence intervals during training. 

%folder = 'G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\';  % You specify this!
%fullMatFileName = fullfile(folder,  'confidence_intervals_taus.mat');
%if ~exist(fullMatFileName, 'file')
%  message = sprintf('%s does not exist', fullMatFileName);
%  uiwait(warndlg(message));
%else
%  confidence_intervals_for_significant_taus = load(fullMatFileName);
%end
%confidence_intervals_for_significant_taus = struct2cell(confidence_intervals_for_significant_taus);

%Indexes(rows and colums) for non-zero Normalized IFs
%[row,col] = find(taus);

%The bounds for phi
%lb = 0;
%ub = 0.999;
%for i=1:height(row)
%    lb = [lb;confidence_intervals_for_significant_taus{1, 1}{row(i),col(i)}(1)];
%    ub = [ub;confidence_intervals_for_significant_taus{1, 1}{row(i),col(i)}(2)];
%end

options = optimoptions('particleswarm','SwarmSize',100,'HybridFcn',@fmincon, 'Display', 'iter');

rng default  % For reproducibility
[x,fval,exitflag,output] = particleswarm(CostFunction,nvars,lb,ub,options)

filepath_for_best_solution = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\best_solution.csv");
writematrix(x,filepath_for_best_solution,'Delimiter',',')
