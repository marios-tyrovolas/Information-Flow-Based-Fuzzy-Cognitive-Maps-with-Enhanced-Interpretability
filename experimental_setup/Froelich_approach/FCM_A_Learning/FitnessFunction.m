function z = FitnessFunction(x)
%warning('off')
%The training is performed for each fold

%Fold index
fold = 10;

FCM_outputs=[];

filepath_for_training_X = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\Scaled_X_train_iter_",num2str(fold),".csv");
filepath_for_training_y = append("..\dataset\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\y_train_iter_",num2str(fold),".csv");

X_train = readtable(filepath_for_training_X);
X_train = table2array(X_train);
    
y_train = readtable(filepath_for_training_y);
y_train = table2array(y_train);
    
Simulations_for_all_observations = cell(height(X_train),1);

number_of_input_concepts = width(X_train);
number_of_Activation_Decision_Concepts = width(y_train);

%FCMs parameters initialization

%Candidate Solution
%Parameter g>0 which determines the gain of transformation
g = x(1,1);

%Weight Matrix

candidate_weights = x(1,2:width(x));

W = zeros(number_of_input_concepts+number_of_Activation_Decision_Concepts,number_of_input_concepts+number_of_Activation_Decision_Concepts);
for i=1:number_of_input_concepts
    W(i,number_of_input_concepts+number_of_Activation_Decision_Concepts) = candidate_weights(1,i);
end


%Froelich assumed 1-step reasoning performed by the FCM for each
%observation

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

%Calculate the individual forecasting error for each output concept
ej = abs(y_train-FCM_outputs);

%G(x)
z = 1/(height(X_train)*number_of_Activation_Decision_Concepts) * sum(ej,'all');


end