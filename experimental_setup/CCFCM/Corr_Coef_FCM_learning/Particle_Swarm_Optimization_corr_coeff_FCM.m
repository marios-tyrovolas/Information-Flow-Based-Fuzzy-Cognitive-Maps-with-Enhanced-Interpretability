%%
clc
clear all
warning('off')
%% Particle Swarm Optimization

%Which fold do you want to use?
fold = 10;

CostFunction = @(x) corr_coeff_FCM_cost_function(x); %cost function

%Number of variables
nvars = 1; %phi

% Option1: Causal relations (based on computed IFs) are imposed as constraints in the optimization problem (finding the weights). 
lb = 0;
ub = 0.8;

options = optimoptions('particleswarm','SwarmSize',100,'HybridFcn',@fmincon, 'Display', 'iter');

rng default  % For reproducibility
[x,fval,exitflag,output] = particleswarm(CostFunction,nvars,lb,ub,options)

filepath_for_best_solution = append("..\experimental_setup\CCFCM\Corr_Coef_FCM_learning\optimal_phi_corr_FCM\",num2str(fold),"\corr_coeff_FCM_best_phi_fold_",num2str(fold),".csv");
writematrix(x,filepath_for_best_solution,'Delimiter',',')