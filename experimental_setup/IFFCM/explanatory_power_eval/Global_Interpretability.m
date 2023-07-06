%%
clc
clear all
warning("off")
%% FCMs parameters initialization
fold = 1;

%###############Original X: X matrix before scaling########################
filepath_for_original_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\X_test_iter_",num2str(fold),".csv");
original_X = readtable(filepath_for_original_X);
%##########################################################################

%Near-Optimal Solution
filepath_for_opt_solution = append("..\experimental_setup\IFFCM\near_opt_sol_for_folds\",num2str(fold),"\best_solution_fold_",num2str(fold),".csv");
x = readmatrix(filepath_for_opt_solution);

%Parameter phi to control the nonlinearity of the FCM
phi = x(1,1);

%Weight Matrix
filepath_for_T_matrix = append("..\experimental_setup\IFFCM\IF_FCM_Learning\taus.csv");
taus = readtable(filepath_for_T_matrix);
taus = table2array(taus);
%W = transpose(taus);

%Indexes(rows and colums) for non-zero Normalized IFs
[row,col] = find(taus);

candidate_weights = x(1,2:width(x));

W = zeros(height(taus),width(taus));
for i=1:height(row)
    W(row(i),col(i)) = candidate_weights(1,i);
end

%The weight matrix is the transpose of taus
W=transpose(W);

%Maximum number of iterations during the inference process
T=100;

%Small positive number for the terminating condition of steady state
epsilon = 10^(-5);

number_of_input_concepts = 6;
number_of_Activation_Decision_Concepts = 1;
%% Global Explainability (Feature Importance)

%#########################Degree Centrality################################

degree_centrality = [];
for feature=1:number_of_input_concepts
    %Incoming arcs in()
    in=0;
    for i=1:height(W)
        if W(i,feature)~=0
            in = in + 1;
        end
    end
    
    out=0;
    %Incoming arcs out()
    for j=1:width(W)
        if W(feature,j)~=0
            out = out + 1;
        end
    end
    degree_centrality(feature,1) = in + out;
end

bar(degree_centrality);
title('Importance Estimates');
ylabel('Estimates');
xlabel('Features');
h = gca;
h.XTickLabel = original_X.Properties.VariableNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';