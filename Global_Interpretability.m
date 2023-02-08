%%
clc
clear all
warning("off")
%% FCMs parameters initialization
fold = 1;

%###############Original X: X matrix before scaling########################
filepath_for_original_X = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(fold),"\Test Dataset","\X_test_iter_",num2str(fold),".csv");
original_X = readtable(filepath_for_original_X);
%##########################################################################

%Near-Optimal Solution
filepath_for_opt_solution = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\near_opt_sol_for_folds\",num2str(fold),"\best_solution_with_napoles_error_function.csv");
x = readmatrix(filepath_for_opt_solution);

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

%#########################Rotshtein's Algorithm############################

%Specifying the initial vectors
Initial_vectors = [];
first_vector = [1,zeros(1,number_of_input_concepts-1),zeros(1,number_of_Activation_Decision_Concepts)];
for i=1:(number_of_input_concepts+number_of_Activation_Decision_Concepts-1)
    Initial_vectors = [Initial_vectors;circshift(first_vector,i-1)];
end

Importance_Indexes=zeros(height(Initial_vectors),1);

for k=1:height(Initial_vectors)
    disp(append("#",num2str(k)," observation"));

    %Inference matrix for each observation
    A=[];
    A(1,:) = Initial_vectors(k,:);

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
    
    Importance_Indexes(k,1)=A(height(A),(number_of_input_concepts+1):width(A));
end

figure;
bar(Importance_Indexes);
title('Importance Estimates');
ylabel('Estimates');
xlabel('Features');
h = gca;
h.XTickLabel = original_X.Properties.VariableNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';