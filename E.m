function z = E(x)
%warning('off')
%The training is performed for each fold

FCM_outputs=[];
Predicted_classes=[];

%FCMs parameters initialization
syms xi

%Candidate Solution
%Parameter phi to control the nonlinearity of the FCM
phi = x(1,1);

%Weight Matrix
filepath_for_T_matrix = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\taus.csv");
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

%Fold index
fold = 1;

filepath_for_training_X = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\Scaled_X_train_iter_",num2str(fold),".csv");
filepath_for_training_y = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Training Dataset","\y_train_iter_",num2str(fold),".csv");

X_train = readtable(filepath_for_training_X);
X_train = table2array(X_train);
    
y_train = readtable(filepath_for_training_y);
y_train = table2array(y_train);
    
Simulations_for_all_folds = cell(height(X_train),fold);

number_of_input_concepts = width(X_train);
number_of_Activation_Decision_Concepts = width(y_train);

%Inference process for each observation

for k=1:height(X_train)
    %disp(append("#",num2str(k)," observation"));
        
    %Inference matrix for each observation (i.e., data instance)
    A=[];
    
    %Initial Stimuli
    A(1,:) = [X_train(k,:),zeros(1,width(y_train))];
    
    %Qausi nonlinear reasoning rule
    A(2,:) = phi.*((A(1,:)*W)./norm(A(1,:)*W,2))+(1-phi).*A(1,:);
    
        
    abs_diff=[];
    %Check if the FCM is in the steady state
    abs_diff = abs(A(2,:)-A(1,:));
    
    each_element=1:width(A(2,:));
  
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
        
        each_element=1:width(A(i,:));

        if all(abs_diff(each_element) < epsilon)
            condition = true;
        else
            condition =false;
        end
        i=i+1;
    end
    
    Simulations_for_all_folds{k,fold} = A;
    
    %FCM output(s)
    FCM_outputs(k,:)=A(height(A),7:width(A));
       
end



%Calculate the individual forecasting error for each output concept
Error = (y_train-FCM_outputs).^2;

%G(x)
G = 1/(height(X_train)*number_of_Activation_Decision_Concepts) * sum(Error,'all');

%H(x)
H=0;
for k=1:height(X_train)
    for i=1:(number_of_input_concepts+number_of_Activation_Decision_Concepts)
        for t=2:height(Simulations_for_all_folds{k,fold})
            H = H + (2*(t/T)*(Simulations_for_all_folds{k,fold}(t,i)-Simulations_for_all_folds{k,fold}(t-1,i))^2);
        end 
    end
end

H = H/(height(X_train)*(number_of_input_concepts+number_of_Activation_Decision_Concepts)*(T-1));

a1=0.8;
a2=1-a1;
z = (a1*G)+(a2*H);

end