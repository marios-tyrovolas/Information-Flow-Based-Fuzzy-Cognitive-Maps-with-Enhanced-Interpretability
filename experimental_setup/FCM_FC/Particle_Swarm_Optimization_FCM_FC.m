%%
clc
clear all
warning('off')
%% Particle Swarm Optimization

%Which fold do you want to use?
fold = 10;

CostFunction = @(x) fcm_fc_training_cost_function(x); %cost function

%#########################################################################
filepath_for_training_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\Scaled_X_train_iter_",num2str(fold),".csv");
filepath_for_training_y = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\y_train_iter_",num2str(fold),".csv");

X_train = readtable(filepath_for_training_X);
X_train = table2array(X_train);

y_train = readtable(filepath_for_training_y);
y_train = table2array(y_train);
%#########################################################################


%Number of variables to be optimized
number_of_variables = width(X_train)+width(y_train);
nvars = ((number_of_variables^2)-number_of_variables)+1; %all weights+phi

lb = [0;ones((number_of_variables^2-number_of_variables),1)*(-1)];
ub = [0.8;ones((number_of_variables^2-number_of_variables),1)*(1)];

options = optimoptions('particleswarm','UseParallel', true,'SwarmSize',100,'HybridFcn',@fmincon, 'Display', 'iter');

rng default  % For reproducibility
[x,fval,exitflag,output] = particleswarm(CostFunction,nvars,lb,ub,options)

%Save the near-optimal solution
filepath_for_best_solution = append("..\experimental_setup\FCM_FC\fcm_fc_near_opt_sol_folds\",num2str(fold),"\best_solution_fcm_fc_fold_",num2str(fold),".csv");
writematrix(x,filepath_for_best_solution,'Delimiter',',')