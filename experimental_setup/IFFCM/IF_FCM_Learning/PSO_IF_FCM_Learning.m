%%
clc
clear all
warning('off')
%% Particle Swarm Optimization

%Which fold do you want to use?
fold = 10;

CostFunction = @(x) cost_function(x); %cost function

%Number of variables
filepath_for_T_matrix = append("..\experimental_setup\IFFCM\IF_FCM_Learning\taus.csv");
taus = readtable(filepath_for_T_matrix);
taus = table2array(taus);
N = nnz(taus); %number of non-zero taus
nvars = N+1; %N+phi

% Causal relations (based on computed IFs) are imposed as constraints in the optimization problem (finding the weights). 
lb = [0;-1;-1;-1;-1;-1;-1;-1;-1;-1];
ub = [0.8;1;1;1;1;1;1;1;1;1];

options = optimoptions('particleswarm','SwarmSize',100,'HybridFcn',@fmincon, 'Display', 'iter');

rng default  % For reproducibility
[x,fval,exitflag,output] = particleswarm(CostFunction,nvars,lb,ub,options)

filepath_for_best_solution = append("..\experimental_setup\IFFCM\near_opt_sol_for_folds\",num2str(fold),"\best_solution_fold_",num2str(fold),".csv");
writematrix(x,filepath_for_best_solution,'Delimiter',',')