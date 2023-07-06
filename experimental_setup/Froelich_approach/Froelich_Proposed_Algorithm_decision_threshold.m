%%
clc
clear all
warning("off")
%% Froelich Proposed Algorithm 

%Input:
%-Actual and predicted states of the FCM output concept for the learning dataset 
%m-Number of classes to be discriminated, 
%d-Minimal distance between the thresholds(accuracy of the search), 
%a1<a2<am- Numerical representatives of the class labels.

%Output:
%Set of thresholds tr0<tr2<…<trm.

%Original dataset
ai4i2020 = readtable("..\dataset\raw_data\ai4i2020_encoded_balanced.csv");
ai4i2020 = table2array(ai4i2020);

%#########################################################################

%Fold index
fold = 10;

FCM_outputs=[];

%###############Original X: X matrix before scaling########################
filepath_for_original_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\X_test_iter_",num2str(fold),".csv");
original_X = readtable(filepath_for_original_X);
original_X = table2array(original_X);
%##########################################################################


filepath_for_X_train = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\Scaled_X_train_iter_",num2str(fold),".csv");
filepath_for_y_train = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\y_train_iter_",num2str(fold),".csv");

filepath_for_X_test = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\Scaled_X_test_iter_",num2str(fold),".csv");
filepath_for_y_test = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\y_test_iter_",num2str(fold),".csv");

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
filepath_for_opt_solution = append("..\experimental_setup\Froelich_approach\near_opt_sol_for_folds\",num2str(fold),"\froelich_best_solution_fold_",num2str(fold),".csv");
x = readmatrix(filepath_for_opt_solution);

%Parameter g>0 which determines the gain of transformation
g = x(1,1);

%Weight Matrix

candidate_weights = x(1,2:width(x));

W = zeros(number_of_input_concepts+number_of_Activation_Decision_Concepts,number_of_input_concepts+number_of_Activation_Decision_Concepts);
for i=1:number_of_input_concepts
    W(i,number_of_input_concepts+number_of_Activation_Decision_Concepts) = candidate_weights(1,i);
end

%Froelich assumed 1-step reasoning performed by the FCM for each
%training observation

for k=1:height(X_train)
    %disp(append("#",num2str(k)," observation"));
        
    %Inference matrix for each observation (i.e., data instance)
    A=[];
    
    %Initial Stimuli
    A(1,:) = [X_train(k,:),zeros(1,width(y_train))];
    
    %Classic reasoning rule
    A(2,:) = sigmf((A(1,:)*W),[g 0]);
    
    Simulations_for_all_observations{k,1} = A;
    
    %FCM output(s)
    FCM_outputs(k,:)=A(height(A),(number_of_input_concepts+1):width(A));
       
end

%% Algorithm 1: The thresholds that are obtained (from the training set) are used for the classification of new data instances (Test set).

%Algorithm initialization
m = 2; %number of classes
d = 0.01; %search accuracy

tr0 = 0; %first threshold, lower bound of the searched interval
trm = 1; %last threshold, upper bound of the searched interval

trk_minus_1 = tr0;
error_history = [];
index = 1;

%Numerical representatives of class labels
a = [];
for k=1:m
    a(k,1) = (k-1)/(m-1);
end

sensitivity = [];
specificity = [];
Thresholds = [];
i = 1;


%searching for the threshold trk
for k=1:m-1
    %errmin = 1; %temporary number of the classification errors
    errmin = height(X_train);
    %The first possible value for the threshold trk
    tr = trk_minus_1 + d;

    % a loop related to the search of the [0,1] interval
    while tr < 1
        % a loop related to the data instances
        err=0; %current number of classification errors
        for r=1:height(X_train)
            if (((FCM_outputs(r,1)>=trk_minus_1) && (FCM_outputs(r,1)<tr)) && (y_train(r,1) ~= a(k,1)))  %|| (((FCM_outputs(r,1)<trk_minus_1) || (FCM_outputs(r,1)>=tr)) && (y_train(r,1) == a(k,1))) % Consider also the False Positives
                err = err + 1;
            end
        end
        
        %Record the error for each threshold
        error_history(index,1) = err;
        index = index + 1;
        
        % calculate the lower bound of tr
        if err < errmin
            errmin = err;
            trl = tr;
        end
        %calculate the upper bound of tr
        if err == errmin
            tru=tr;
        end
        
        %Predict he outputs for each examined threshold
        for deiktis=1:height(X_train)
            if FCM_outputs(deiktis,1)>=tr
                Predicted_classes(deiktis,1)=1;
            else
                Predicted_classes(deiktis,1)=0;
            end
        end
        
        %Calculate the sensitivity and specificity for each examined
        %threshold
        metrics = Evaluate(y_train,Predicted_classes);
        sensitivity(i,1) = metrics(1,2);
        specificity(i,1) = metrics(1,3);
        
        Thresholds(i,1) = tr;
        i = i + 1;
        
        tr = tr + d; % the next threshold value to check
    end
    %calculate trk
    trk = (tru+trl)/2;
    trk_minus_1 = trk;
end

%% 

%Froelich assumed 1-step reasoning performed by the FCM for each
%test observation

for k=1:height(X_test)
    %disp(append("#",num2str(k)," observation"));
        
    %Inference matrix for each observation (i.e., data instance)
    A=[];
    
    %Initial Stimuli
    A(1,:) = [X_test(k,:),zeros(1,width(y_test))];
    
    %Classic reasoning rule
    A(2,:) = sigmf((A(1,:)*W),[g 0]);
    
    Simulations_for_all_observations_test{k,1} = A;
    
    %FCM output(s)
    FCM_outputs_test(k,:)=A(height(A),(number_of_input_concepts+1):width(A));
       
end


%Th:Thresholds on classifier scores
%OPTROCPT — Optimal operating point of the ROC curve
[fpr,tpr,Th,AUC_test,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(y_test,FCM_outputs_test,1);

for k=1:height(X_test)
    if FCM_outputs_test(k,1)>=trk
        Predicted_classes_test(k,1)=1;
    else
        Predicted_classes_test(k,1)=0;
    end
end

%Evaluation metrics for the calculated threshold
EVAL = Evaluate(y_test,Predicted_classes_test);

fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n',[EVAL,cohensKappa(y_test, Predicted_classes_test)]);

figure;
subplot(2,1,1)
plot(Thresholds,error_history)
xlabel('Threshold') 
ylabel('Error')
title('Error curve')
hold on
plot(trk,errmin,'r*')

subplot(2,1,2)
plot(Thresholds,sensitivity,Thresholds,specificity)
hold on 
xline(trk,'m--',num2str(trk),'LineWidth',1.5)
xlabel('Threshold') 
title('Sensitivity and Specificity')
legend("Sensitivity","Specificity")