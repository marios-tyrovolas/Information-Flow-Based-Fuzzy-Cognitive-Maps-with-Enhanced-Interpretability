%%
clc
clear all
warning("off")
%Libraries
addpath '..\extra_libraries\Evaluate'
addpath '..\extra_libraries\kappa_value'
%% FCM Expert Topology Inference process
%Class-per-output (CpO) architecture - in this case, the class attribute
%of the dataset is mapped to several of FCM's concepts with one
%concept for each class. The concept with the highest activation value in
%the last iteration indicates the predicted class.

%Which is the FCM's density
density =20;

%Which topology for this specific density
topology_try = 2;

%Which fold do you want to use?
fold = 10;

%Number of folds for stratified k-fold cross validation
numbfolds=9;

%Original dataset
ai4i2020 = readtable("..\dataset\raw_data\ai4i2020_encoded_balanced.csv");
ai4i2020 = table2array(ai4i2020);

%#########################################################################

%FCMs parameters initialization

%Weight Matrix
filepath_for_fcm_weights = append("..\experimental_setup\FCM_Expert_approach\FCM_clf_all_param_optim\Density_",num2str(density),"\Try_",num2str(topology_try),"\solut_folds\",num2str(fold),"\FCM_clf_fold_",num2str(fold),".csv");
fcm_weights = readmatrix(filepath_for_fcm_weights);

%Parameter phi to control the nonlinearity of the FCM
W = fcm_weights(2:end,2:end);

%Maximum number of iterations during the inference process
T=20;

%Small positive number for the terminating condition of steady state
epsilon = 0.001;

disp(append(newline,num2str(fold)," fold"));

filepath_for_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\Scaled_X_test_iter_",num2str(fold),".csv");
filepath_for_y = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\y_test_iter_",num2str(fold),".csv");

X = readtable(filepath_for_X);
X = table2array(X);

y = readtable(filepath_for_y);
y = table2array(y);

%###############Original X: X matrix before scaling########################
filepath_for_original_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\X_test_iter_",num2str(fold),".csv");
original_X = readtable(filepath_for_original_X);
original_X = table2array(original_X);
%##########################################################################

Simulations_for_all_observations = cell(height(X),1);
Outputs=zeros(height(X),2);
Predicted_classes=zeros(height(X),1);
Predicted_classes_2=zeros(height(X),1);
Predicted_classes_3=zeros(height(X),1);
Predicted_classes_4=zeros(height(X),1);

number_of_input_concepts = width(X);
number_of_Activation_Decision_Concepts = length(unique(y));

%Inference process for each observation

for k=1:height(X)
    disp(append("#",num2str(k)," observation"));

    %Inference matrix for each observation
    A=[];
    A(1,:) = [X(k,:),zeros(1,number_of_Activation_Decision_Concepts)];

    %Kosko's activation rule with self-memory
     A(2,:) = sigmf((A(1,:)*W+A(1,:)),[10 1]);

    %If any neuron has different sigmoid parameters
    %##############################################
    %A(2,1) = sigmf((A(1,:)*W(:,1)+A(1,1)),[6.973 0.681]); %ty
    %A(2,2) = sigmf((A(1,:)*W(:,2)+A(1,2)),[5.875 0.763]); %at
    %A(2,3) = sigmf((A(1,:)*W(:,3)+A(1,3)),[7.922 0.542]); %pt
    %A(2,4) = sigmf((A(1,:)*W(:,4)+A(1,4)),[7.788 0.659]); %rs
    %A(2,5) = sigmf((A(1,:)*W(:,5)+A(1,5)),[5.909 0.998]); %tq
    %A(2,6) = sigmf((A(1,:)*W(:,6)+A(1,6)),[8.578 0.772]); %tw
    %A(2,7) = sigmf((A(1,:)*W(:,7)+A(1,7)),[6.289 0.496]); %1
    %A(2,8) = sigmf((A(1,:)*W(:,8)+A(1,8)),[4.044 0.452]); %0
    %##############################################

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

        %Kosko's activation rule with self-memory
        A(i,:) = sigmf((A(i-1,:)*W)+A(i-1,:),[10 1]);
        
        %If any neuron has different sigmoid parameters
        %##############################################
        %A(i,1) = sigmf((A(i-1,:)*W(:,1)+A(i-1,1)),[6.973 0.681]); %ty
        %A(i,2) = sigmf((A(i-1,:)*W(:,2)+A(i-1,2)),[5.875 0.763]); %at
        %A(i,3) = sigmf((A(i-1,:)*W(:,3)+A(i-1,3)),[7.922 0.542]); %pt
        %A(i,4) = sigmf((A(i-1,:)*W(:,4)+A(i-1,4)),[7.788 0.659]); %rs
        %A(i,5) = sigmf((A(i-1,:)*W(:,5)+A(i-1,5)),[5.909 0.998]); %tq
        %A(i,6) = sigmf((A(i-1,:)*W(:,6)+A(i-1,6)),[8.578 0.772]); %tw
        %A(i,7) = sigmf((A(i-1,:)*W(:,7)+A(i-1,7)),[6.289 0.496]); %1
        %A(i,8) = sigmf((A(i-1,:)*W(:,8)+A(i-1,8)),[4.044 0.452]); %0
        %##############################################

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
   Outputs(k,:)=A(height(A),(number_of_input_concepts+1):width(A));
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


%The concept with the highest activation value indicates the predicted class.
for k=1:height(X)
    if Outputs(k,1)==max(Outputs(k,:))
        Predicted_classes(k,1)=1;
    else
        Predicted_classes(k,1)=0;
    end
end

%% Calculate roc curves

%Evaluation metrics
EVAL = Evaluate(y,Predicted_classes);

fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n\n',[EVAL,cohensKappa(y, Predicted_classes)]);

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

denominator = 0;
        
if number_of_TWF_failures_in_true_positive_predictions ~= 0
    twf_success = correct_explanations_TWF/number_of_TWF_failures_in_true_positive_predictions;
    denominator = denominator + 1;
else
    twf_success = 0;
end
if number_of_HDF_failures_in_true_positive_predictions ~= 0
    hdf_success = correct_explanations_HDF/number_of_HDF_failures_in_true_positive_predictions;
    denominator = denominator + 1;
else
    hdf_success = 0;
end
if number_of_PWF_failures_in_true_positive_predictions ~= 0
    pwf_success = correct_explanations_PWF/number_of_PWF_failures_in_true_positive_predictions;
    denominator = denominator + 1;
else
    pwf_success = 0;
end
if number_of_OSF_failures_in_true_positive_predictions ~= 0
    osf_success = correct_explanations_OSF/number_of_OSF_failures_in_true_positive_predictions;
    denominator = denominator + 1;
else
    osf_success = 0;
end

average_success = (twf_success+hdf_success+pwf_success+osf_success)/denominator;

fprintf("TWF: %.4f success, HDF: %.4f success, PWF: %.4f success, OSF: %.4f success \n\n",correct_explanations_TWF/number_of_TWF_failures_in_true_positive_predictions,correct_explanations_HDF/number_of_HDF_failures_in_true_positive_predictions,correct_explanations_PWF/number_of_PWF_failures_in_true_positive_predictions,correct_explanations_OSF/number_of_OSF_failures_in_true_positive_predictions);

%average_success = ((correct_explanations_TWF/number_of_TWF_failures_in_true_positive_predictions)+(correct_explanations_HDF/number_of_HDF_failures_in_true_positive_predictions)+(correct_explanations_PWF/number_of_PWF_failures_in_true_positive_predictions)+(correct_explanations_OSF/number_of_OSF_failures_in_true_positive_predictions))/4;

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