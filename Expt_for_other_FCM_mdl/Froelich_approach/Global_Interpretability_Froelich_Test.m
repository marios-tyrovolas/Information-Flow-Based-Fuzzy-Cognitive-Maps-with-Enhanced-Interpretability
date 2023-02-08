%%
clc
clear all
warning("off")
%% FCM parameters

%Fold index
fold = 10;

FCM_outputs=[];

%###############Original X: X matrix before scaling########################
filepath_for_original_X = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\X_test_iter_",num2str(fold),".csv");
original_X = readtable(filepath_for_original_X);
original_X = table2array(original_X);
%##########################################################################


filepath_for_X_train = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\Scaled_X_train_iter_",num2str(fold),".csv");
filepath_for_y_train = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\y_train_iter_",num2str(fold),".csv");

filepath_for_X_test = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\Scaled_X_test_iter_",num2str(fold),".csv");
filepath_for_y_test = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\y_test_iter_",num2str(fold),".csv");

X_train = readtable(filepath_for_X_train);
X_train = table2array(X_train);
    
y_train = readtable(filepath_for_y_train);
y_train = table2array(y_train);

X_test = readtable(filepath_for_X_test);
X_test = table2array(X_test);
    
y_test = readtable(filepath_for_y_test);
y_test = table2array(y_test);
    
Simulations_for_all_observations = cell(height(X_train),1);
Simulations_for_all_observations_test = cell(height(X_test),1);

number_of_input_concepts = width(X_train);
number_of_Activation_Decision_Concepts = width(y_train);

%Near-Optimal Solution
filepath_for_opt_solution = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\Expt_for_other_FCM_mdl\Froelich_approach\near_opt_sol_for_folds\",num2str(fold),"\froelich_best_solution_fold_",num2str(fold),".csv");
x = readmatrix(filepath_for_opt_solution);

%Parameter g>0 which determines the gain of transformation
g = x(1,1);

%Weight Matrix

candidate_weights = x(1,2:width(x));

W = zeros(number_of_input_concepts+number_of_Activation_Decision_Concepts,number_of_input_concepts+number_of_Activation_Decision_Concepts);
for i=1:number_of_input_concepts
    W(i,number_of_input_concepts+number_of_Activation_Decision_Concepts) = candidate_weights(1,i);
end
%% Rotshtein's Algorithm

%Specifying the initial vectors
Initial_vectors = [];
first_vector = [1,zeros(1,number_of_input_concepts-1),zeros(1,number_of_Activation_Decision_Concepts)];
for i=1:(number_of_input_concepts+number_of_Activation_Decision_Concepts-1)
    Initial_vectors = [Initial_vectors;circshift(first_vector,i-1)];
end

Importance_Indexes=zeros(height(Initial_vectors),1);

for k=1:height(Initial_vectors)
    disp(append("#",num2str(k)," observation"));

    %Inference matrix for each observation (i.e., data instance)
    A=[];
    
    %Initial Stimuli
    A(1,:) = [X_train(k,:),zeros(1,width(y_train))];
    
    %Classic reasoning rule
    A(2,:) = sigmf((A(1,:)*W),[g 0]);
    
    Importance_Indexes(k,1)=A(height(A),(number_of_input_concepts+1):width(A));
end

figure;
bar(Importance_Indexes);
title('Importance Estimates');
ylabel('Estimates');
xlabel('Features');
h = gca;