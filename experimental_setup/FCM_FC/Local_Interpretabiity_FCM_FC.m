%%
clc
clear all
warning("off")
%Libraries
addpath '..\extra_libraries\Evaluate'
addpath '..\extra_libraries\kappa_value'
%% 

%Which fold do you want to use?
fold = 10;

%Number of folds for stratified k-fold cross validation
numbfolds=10;

%Original dataset
ai4i2020 = readtable("..\dataset\raw_data\ai4i2020_encoded_balanced.csv");
ai4i2020 = table2array(ai4i2020);

%#########################################################################

disp(append(newline,num2str(fold)," fold"));

filepath_for_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\Scaled_X_test_iter_",num2str(fold),".csv");
filepath_for_y = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\y_test_iter_",num2str(fold),".csv");

X = readtable(filepath_for_X);
X = table2array(X);

y = readtable(filepath_for_y);
y = table2array(y);

%#########################################################################

%FCMs parameters initialization

%Near-Optimal Solution
filepath_for_opt_solution = append("..\experimental_setup\FCM_FC\fcm_fc_near_opt_sol_folds\",num2str(fold),"\best_solution_fcm_fc_fold_",num2str(fold),".csv");
x = readmatrix(filepath_for_opt_solution);

%Parameter phi to control the nonlinearity of the FCM
phi = x(1,1);

%Weight Matrix

number_of_variables = width(X)+width(y);

candidate_weights = x(1,2:width(x));

%Declare a matrix full of ones except on the diagonal
W=ones(number_of_variables,number_of_variables);
W=W-diag(diag(W));
%Replace all the ones by the desired off-diagonal values
W(W==1)=candidate_weights;

%Maximum number of iterations during the inference process
T=100;

%Small positive number for the terminating condition of steady state
epsilon = 10^(-5);


%###############Original X: X matrix before scaling########################
filepath_for_original_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\X_test_iter_",num2str(fold),".csv");
original_X = readtable(filepath_for_original_X);
original_X = table2array(original_X);
%##########################################################################

Simulations_for_all_observations = cell(height(X),1);
Outputs=zeros(height(X),1);
Predicted_classes=zeros(height(X),1);
Predicted_classes_2=zeros(height(X),1);
Predicted_classes_3=zeros(height(X),1);
Predicted_classes_4=zeros(height(X),1);

number_of_input_concepts = width(X);
number_of_Activation_Decision_Concepts = width(y);

%Inference process for each observation

for k=1:height(X)
    disp(append("#",num2str(k)," observation"));

    %Inference matrix for each observation
    A=[];
    A(1,:) = [X(k,:),zeros(1,width(y))];

    %Qausi nonlinear reasoning rule
     A(2,:) = phi.*((A(1,:)*W)./norm(A(1,:)*W,2))+(1-phi).*A(1,:);


    abs_diff=[];
    %Check if the FCM is in the steady state
    abs_diff = abs(A(2,:)-A(1,:));

    each_element=(1:width(A(2,:)));

    if all(abs_diff(each_element) < epsilon)
        condition = true;
    else
        condition =false;
    end

    %Until the FCM converge to the equilibrium point or the maximum number
    %of iterations T is reached
    i=3;
    while (condition ~= true) && (i<=T)

        %Qausi nonlinear reasoning rule
        A(i,:) = phi.*((A(i-1,:)*W)./norm(A(i-1,:)*W,2))+(1-phi).*A(1,:);

        %Check if the FCM is in the steady state
        abs_diff = abs(A(i,:)-A(i-1,:));

        each_element=(1:width(A(i,:)));

        if all(abs_diff(each_element) < epsilon)
            condition = true;
        else
            condition =false;
        end
        i=i+1;
    end

   Simulations_for_all_observations{k,1} = A;       
   Outputs(k,1)=A(height(A),(number_of_input_concepts+1):width(A)); 
end

counter=0;
for observation=1:height(X)
    if height(Simulations_for_all_observations{observation,1}) < T
        counter=counter+1;
    end
end

if counter==height(X)
    disp("The FCM converges to steady state for each data instance")
else
    fprintf("\nAt %d data obesrvation(s) the FCM has cyclic or chaotic behavior\n\n",height(X)-counter)
end

%Gamma measure
Gamma=0;
for k=1:height(X)
    for i=1:(number_of_input_concepts+number_of_Activation_Decision_Concepts)
        for t=2:height(Simulations_for_all_observations{k,1})
            Gamma = Gamma + (Simulations_for_all_observations{k,1}(t,i)-Simulations_for_all_observations{k,1}(t-1,i))^2;
        end 
    end
end

Gamma = Gamma/(height(X)*(number_of_input_concepts+number_of_Activation_Decision_Concepts)*(T-1));

fprintf("Gamma measure: %.4f\n\n",Gamma)


%% Calculate roc curves

%Th:Thresholds on classifier scores
%OPTROCPT — Optimal operating point of the ROC curve
[fpr,tpr,Th,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(y,Outputs,1);

%Calculate the precision for each threshold
[prec,tpr1,Th1] = perfcurve(y,Outputs,1,'XCrit','prec');

plot(fpr,tpr)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC curve for Classification by FCM')
hold on
plot(OPTROCPT(1),OPTROCPT(2),'r*')

%calculate the g-mean for each threshold
gmeans = sqrt(tpr.*(1-fpr));

%locate the index of the largest g-mean
[~,idx] = max(gmeans);
best_thresh_1 = Th(idx);

fprintf('Best Threshold=%f, G-Mean=%.4f \n', best_thresh_1, gmeans(idx));

%Alternative metric:  Youden’s J statistic
%get the best threshold
J = tpr - fpr;
[~,idxJ] = max(J);
best_thresh_2 = Th(idxJ);
fprintf('Best Threshold according to Youden’s J statistic=%f\n',best_thresh_2);


%Alternative metric: Find the threshold that corresponds to the optimal operating point.
best_thresh_3 = Th((fpr==OPTROCPT(1))&(tpr==OPTROCPT(2)));


%Alternative metric: locate the index of the largest f score

%Step1: convert to f score
reca = tpr;
fscore = (2*(prec.*reca))./(prec+reca);
[~,idxfsc] = max(fscore);
best_thresh_4 = Th(idxfsc);
fprintf('Best Threshold=%f, F-Score=%.4f\n\n', best_thresh_4, fscore(idxfsc));

for k=1:height(X)
    if Outputs(k,1)>=best_thresh_1
        Predicted_classes(k,1)=1;
    else
        Predicted_classes(k,1)=0;
    end
end

%Evaluation metrics for the first optimal threshold
EVAL = Evaluate(y,Predicted_classes);

fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n\n',[EVAL,cohensKappa(y, Predicted_classes)]);

if best_thresh_2~=best_thresh_1
    for k=1:height(X)
        if Outputs(k,1)>=best_thresh_2
            Predicted_classes_2(k,1)=1;
        else
            Predicted_classes_2(k,1)=0;
        end
    end
   %Evaluation metrics for the second optimal threshold
   EVAL_2 = Evaluate(y,Predicted_classes_2);
   
   fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n\n',[EVAL_2,cohensKappa(y, Predicted_classes_2)]);
end

if (best_thresh_3~=best_thresh_2) && (best_thresh_3~=best_thresh_1)
    for k=1:height(X)
        if Outputs(k,1)>=best_thresh_3
            Predicted_classes_3(k,1)=1;
        else
            Predicted_classes_3(k,1)=0;
        end
    end
   %Evaluation metrics for the second optimal threshold
   EVAL_3 = Evaluate(y,Predicted_classes_3);
   
   fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n\n',[EVAL_3,cohensKappa(y, Predicted_classes_3)]);
end

if (best_thresh_4~=best_thresh_1) && (best_thresh_4~=best_thresh_2) && (best_thresh_4~=best_thresh_3)
    for k=1:height(X)
        if Outputs(k,1)>=best_thresh_4
            Predicted_classes_4(k,1)=1;
        else
            Predicted_classes_4(k,1)=0;
        end
    end
   %Evaluation metrics for the second optimal threshold
   EVAL_4 = Evaluate(y,Predicted_classes_4);
   
   fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n\n',[EVAL_4,cohensKappa(y, Predicted_classes_4)]);
end

%% Local Feature Importance

%Find the observations' indexes that the FCM predicted correctly as faulty
true_positive_indexes = [];
i=1;
for k=1:height(y)
    if (y(k,1)==Predicted_classes(k,1)) && (y(k,1)==1)
        true_positive_indexes(i,1)=k;
        i=i+1;
    end
end

%Calculate the feature importance for the above observations
local_feature_importance = cell(height(true_positive_indexes),1);
for k=1:height(true_positive_indexes)
    local_feature_importance{k,:} = abs(Simulations_for_all_observations{true_positive_indexes(k,1),1}(height(Simulations_for_all_observations{true_positive_indexes(k,1),1}),1:number_of_input_concepts));
end

%for k=1:height(true_positive_indexes)
%    local_feature_importance{k,:} = phi.*(Simulations_for_all_observations{true_positive_indexes(k,1),1}(height(Simulations_for_all_observations{true_positive_indexes(k,1),1})-1,1:number_of_input_concepts).*transpose(W(1:6,7)))./norm(Simulations_for_all_observations{true_positive_indexes(k,1),1}(height(Simulations_for_all_observations{true_positive_indexes(k,1),1})-1,:)*W,2);
%end

%Find the observations that the FCM correctly predicted as faulty based on
%the previous indexes
true_positive_observations=[];
for k=1:height(true_positive_indexes)
    true_positive_observations(k,:) = [original_X(true_positive_indexes(k,1),:),y(true_positive_indexes(k,1),:)];
end

%Find the observations that the FCM predicted correctly as faulty along with
%the failure modes

true_positive_observations_with_failure_modes = [];
for index1=1:height(true_positive_observations)
    for index2=1:height(ai4i2020)
        if isequal(true_positive_observations(index1,:),ai4i2020(index2,1:7))==true
            true_positive_observations_with_failure_modes = [true_positive_observations_with_failure_modes;ai4i2020(index2,:)];
        end
    end
end


number_of_TWF_failures_in_true_positive_predictions = 0;
number_of_HDF_failures_in_true_positive_predictions = 0;
number_of_PWF_failures_in_true_positive_predictions = 0;
number_of_OSF_failures_in_true_positive_predictions = 0;
number_of_random_failures_in_true_positive_predictions=0;

for k=1:height(true_positive_observations_with_failure_modes)
    if true_positive_observations_with_failure_modes(k,8)==1 %if the failure mode is TWF
        number_of_TWF_failures_in_true_positive_predictions = number_of_TWF_failures_in_true_positive_predictions + 1;
    end
    if true_positive_observations_with_failure_modes(k,9)==1 %if the failure mode is HDF
        number_of_HDF_failures_in_true_positive_predictions = number_of_HDF_failures_in_true_positive_predictions + 1;
    end
    if true_positive_observations_with_failure_modes(k,10)==1 %if the failure mode is PWF
        number_of_PWF_failures_in_true_positive_predictions = number_of_PWF_failures_in_true_positive_predictions + 1;
    end
    if true_positive_observations_with_failure_modes(k,11)==1 %if the failure mode is OSF
        number_of_OSF_failures_in_true_positive_predictions = number_of_OSF_failures_in_true_positive_predictions + 1;
    end
    if (true_positive_observations_with_failure_modes(k,7)==1) && (true_positive_observations_with_failure_modes(k,8)==0) && (true_positive_observations_with_failure_modes(k,9)==0) && (true_positive_observations_with_failure_modes(k,10)==0) && (true_positive_observations_with_failure_modes(k,11)==0) && (true_positive_observations_with_failure_modes(k,12)==0)
        number_of_random_failures_in_true_positive_predictions = number_of_random_failures_in_true_positive_predictions+1;
    end
end

%Calculate the correct explanations
correct_explanations = 0;
correct_explanations_TWF = 0;
correct_explanations_HDF = 0;
correct_explanations_PWF = 0;
correct_explanations_OSF = 0;

for k=1:height(true_positive_observations)
    if true_positive_observations_with_failure_modes(k,8)==1 %if the failure mode is TWF
        if local_feature_importance{k,1}(1,6) == max(local_feature_importance{k,1}) %if tool wear is the most important feature
            correct_explanations = correct_explanations+1;
            correct_explanations_TWF = correct_explanations_TWF + 1;
        end
    end
    if true_positive_observations_with_failure_modes(k,9)==1 %if the failure mode is HDF
        temp_matrix = local_feature_importance{k,1};
        [max1, ind1] = max(temp_matrix);
        temp_matrix(ind1) = -Inf;
        [max2, ind2] = max(temp_matrix);
        temp_matrix(ind2) = -Inf;
        if (local_feature_importance{k,1}(1,2) == max1) || (local_feature_importance{k,1}(1,2) == max2) || (local_feature_importance{k,1}(1,3) == max1) || (local_feature_importance{k,1}(1,3) == max2)
            correct_explanations = correct_explanations+1;
            correct_explanations_HDF = correct_explanations_HDF + 1;
        end
    end
    if true_positive_observations_with_failure_modes(k,10)==1 %if the failure mode is PWF
        temp_matrix = local_feature_importance{k,:};
        [max1, ind1] = max(temp_matrix);
        temp_matrix(ind1) = -Inf;
        [max2, ind2] = max(temp_matrix);
        temp_matrix(ind2) = -Inf;
        if (local_feature_importance{k,1}(1,4) == max1) || (local_feature_importance{k,1}(1,4) == max2) || (local_feature_importance{k,1}(1,5) == max1) || (local_feature_importance{k,1}(1,5) == max2)
            correct_explanations = correct_explanations+1;
            correct_explanations_PWF = correct_explanations_PWF + 1;
        end
    end
    if true_positive_observations_with_failure_modes(k,11)==1 %if the failure mode is OSF
        temp_matrix = local_feature_importance{k,:};
        [max1, ind1] = max(temp_matrix);
        temp_matrix(ind1) = -Inf;
        [max2, ind2] = max(temp_matrix);
        temp_matrix(ind2) = -Inf;
        if (local_feature_importance{k,1}(1,6) == max1) || (local_feature_importance{k,1}(1,6) == max2) || (local_feature_importance{k,1}(1,5) == max1) || (local_feature_importance{k,1}(1,5) == max2)
            correct_explanations = correct_explanations+1;
            correct_explanations_OSF = correct_explanations_OSF + 1;
        end
    end          
end

fprintf("TWF: %.4f success, HDF: %.4f success, PWF: %.4f success, OSF: %.4f success \n\n",correct_explanations_TWF/number_of_TWF_failures_in_true_positive_predictions,correct_explanations_HDF/number_of_HDF_failures_in_true_positive_predictions,correct_explanations_PWF/number_of_PWF_failures_in_true_positive_predictions,correct_explanations_OSF/number_of_OSF_failures_in_true_positive_predictions);

average_success = ((correct_explanations_TWF/number_of_TWF_failures_in_true_positive_predictions)+(correct_explanations_HDF/number_of_HDF_failures_in_true_positive_predictions)+(correct_explanations_PWF/number_of_PWF_failures_in_true_positive_predictions)+(correct_explanations_OSF/number_of_OSF_failures_in_true_positive_predictions))/4;

fprintf("Through the FCM, %d correct explanations are made in the true positive predictions with average success: %.4f \n\n", correct_explanations, average_success);
%Plot local feature importance for each true positive prediction
%{
true_positive_observation_I_want_to_see = 1;
figure
barh(local_feature_importance{true_positive_observation_I_want_to_see,:})
yticklabels({'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'})
title_for_plot=append('Local Feature Importance for observation: ', num2str(true_positive_indexes(true_positive_observation_I_want_to_see,1)));
title(title_for_plot)
if true_positive_observations_with_failure_modes(k,8)==1
    annotation('textbox', [0.75, 0.1, 0.1, 0.1], 'String', "TWF")    
elseif true_positive_observations_with_failure_modes(k,9)==1
        annotation('textbox', [0.75, 0.1, 0.1, 0.1], 'String', "HDF")
elseif true_positive_observations_with_failure_modes(k,10)==1
    annotation('textbox', [0.75, 0.1, 0.1, 0.1], 'String', "PWF")
elseif true_positive_observations_with_failure_modes(k,11)==1
    annotation('textbox', [0.75, 0.1, 0.1, 0.1], 'String', "OSF")
else
    annotation('textbox', [0.75, 0.1, 0.1, 0.1], 'String', "RNF")
end
%}