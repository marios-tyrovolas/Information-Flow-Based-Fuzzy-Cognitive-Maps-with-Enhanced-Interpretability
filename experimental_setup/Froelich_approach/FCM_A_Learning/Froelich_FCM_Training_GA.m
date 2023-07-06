%%
clc
clear all
warning('off')
%% Froelich FCM Training Genetic Algorithm

CostFunction = @(x) FitnessFunction(x); %cost function

%Which fold do you want to use?
fold = 10;

%Original dataset
ai4i2020 = readtable("..\dataset\raw_data\ai4i2020_encoded_balanced.csv");
ai4i2020 = table2array(ai4i2020);

%Froelich investigated only the single-output structure of the FCM with a 
%lack of feedback between nodes.
numberOfVariables = 7; %In addition to optimising FCM's weights, we included the gain g within the genotype for this study.


lb = [0;(-1)*ones(width(ai4i2020)-1,1)];
ub = [10;ones(width(ai4i2020)-1,1)];

opts = optimoptions(@ga,'PlotFcn',{@gaplotbestf,@gaplotstopping});

%MaxGenerations: maximal number of generations
%MaxStallGenerations: number of generations without any improvement
%opts = optimoptions(opts,'MaxGenerations',100,'MaxStallGenerations', 10);

%Cardinality of the initial population
%opts.PopulationSize = 10;

%Run the ga solver, including the opts argument.
rng default  % For reproducibility
[x,Fval,exitFlag,Output] = ...
    ga(CostFunction,numberOfVariables,[],[],[],[],lb,ub,[],opts);

filepath_for_best_solution = append("..\experimental_setup\Froelich_approach\near_opt_sol_for_folds\",num2str(fold),"\froelich_best_solution_fold_",num2str(fold),".csv");
writematrix(x,filepath_for_best_solution,'Delimiter',',')