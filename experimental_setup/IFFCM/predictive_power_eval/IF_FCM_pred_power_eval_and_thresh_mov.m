%%
clc
clear all
warning("off")
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

%FCMs parameters initialization

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

disp(append(newline,num2str(fold)," fold"));

filepath_for_training_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\Scaled_X_test_iter_",num2str(fold),".csv");
filepath_for_training_y = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\y_test_iter_",num2str(fold),".csv");

X_train = readtable(filepath_for_training_X);
X_train = table2array(X_train);

y_train = readtable(filepath_for_training_y);
y_train = table2array(y_train);

Simulations_for_all_observations = cell(height(X_train),1);
Outputs=zeros(height(X_train),1);
Predicted_classes=zeros(height(X_train),1);
Predicted_classes_2=zeros(height(X_train),1);
Predicted_classes_3=zeros(height(X_train),1);
Predicted_classes_4=zeros(height(X_train),1);

number_of_input_concepts = width(X_train);
number_of_Activation_Decision_Concepts = width(y_train);

%Inference process for each observation

for k=1:height(X_train)
    disp(append("#",num2str(k)," observation"));

    %Inference matrix for each observation
    A=[];
    A(1,:) = [X_train(k,:),zeros(1,width(y_train))];

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
for observation=1:height(X_train)
    if height(Simulations_for_all_observations{observation,1}) < T
        counter=counter+1;
    end
end

if counter==height(X_train)
    disp("The FCM converges to steady state for each data instance\n\n")
else
    fprintf("\nAt %d data obesrvation(s) the FCM has cyclic or chaotic behavior\n\n",height(X_train)-counter)
end

%Gamma measure
Gamma=0;
for k=1:height(X_train)
    for i=1:(number_of_input_concepts+number_of_Activation_Decision_Concepts)
        for t=2:height(Simulations_for_all_observations{k,1})
            Gamma = Gamma + (Simulations_for_all_observations{k,1}(t,i)-Simulations_for_all_observations{k,1}(t-1,i))^2;
        end 
    end
end

Gamma = Gamma/(height(X_train)*(number_of_input_concepts+number_of_Activation_Decision_Concepts)*(T-1));

fprintf("Gamma measure: %.4f\n\n",Gamma)


%% Calculate roc curves

%Th:Thresholds on classifier scores
%OPTROCPT — Optimal operating point of the ROC curve
[fpr,tpr,Th,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(y_train,Outputs,1);

%Calculate the precision for each threshold
[prec,tpr1,Th1] = perfcurve(y_train,Outputs,1,'XCrit','prec');

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

for k=1:height(X_train)
    if Outputs(k,1)>=best_thresh_1
        Predicted_classes(k,1)=1;
    else
        Predicted_classes(k,1)=0;
    end
end

%Evaluation metrics for the first optimal threshold
EVAL = Evaluate(y_train,Predicted_classes);

fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n',[EVAL,cohensKappa(y_train, Predicted_classes)]);


if best_thresh_2~=best_thresh_1
    for k=1:height(X_train)
        if Outputs(k,1)>=best_thresh_2
            Predicted_classes_2(k,1)=1;
        else
            Predicted_classes_2(k,1)=0;
        end
    end
   %Evaluation metrics for the second optimal threshold
   EVAL_2 = Evaluate(y_train,Predicted_classes_2);
   
   
   fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n',[EVAL_2,cohensKappa(y_train, Predicted_classes_2)]);
end

if (best_thresh_3~=best_thresh_2) && (best_thresh_3~=best_thresh_1)
    for k=1:height(X_train)
        if Outputs(k,1)>=best_thresh_3
            Predicted_classes_3(k,1)=1;
        else
            Predicted_classes_3(k,1)=0;
        end
    end
   %Evaluation metrics for the second optimal threshold
   EVAL_3 = Evaluate(y_train,Predicted_classes_3);
   
   
   fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n',[EVAL_3,cohensKappa(y_train, Predicted_classes_3)]);
end

if (best_thresh_4~=best_thresh_1) && (best_thresh_4~=best_thresh_2) && (best_thresh_4~=best_thresh_3)
    for k=1:height(X_train)
        if Outputs(k,1)>=best_thresh_4
            Predicted_classes_4(k,1)=1;
        else
            Predicted_classes_4(k,1)=0;
        end
    end
   %Evaluation metrics for the second optimal threshold
   EVAL_4 = Evaluate(y_train,Predicted_classes_4);
   
   
   fprintf('[accuracy=%.4f, sensitivity=%.4f, specificity=%.4f, precision=%.4f, recall=%.4f, f_measure=%.4f, gmean=%.4f, kappa=%.4f]\n',[EVAL_4,cohensKappa(y_train, Predicted_classes_4)]);
end


%% Write FCM Predictions (output concept values and crisp labels) in .csv files

if contains(filepath_for_training_X,'Training Dataset')==true
    filepath_for_FCM_ouput_concept_values = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\y_train_pred_fcm.csv");
    writematrix(Outputs,filepath_for_FCM_ouput_concept_values,'Delimiter',',')
    
    filepath_for_FCM_predicted_classes = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\fcm_pred_class_train.csv");
    writematrix(Predicted_classes,filepath_for_FCM_predicted_classes,'Delimiter',',')
    
    disp("Outputs for Training Dataset Saved")
end

if contains(filepath_for_training_X,'Test Dataset')==true
    filepath_for_FCM_ouput_concept_values = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\y_test_pred_fcm.csv");
    writematrix(Outputs,filepath_for_FCM_ouput_concept_values,'Delimiter',',')
    
    filepath_for_FCM_predicted_classes = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\fcm_pred_class_test.csv");
    writematrix(Predicted_classes,filepath_for_FCM_predicted_classes,'Delimiter',',')
    
    disp("Outputs for Test Dataset Saved")
end


