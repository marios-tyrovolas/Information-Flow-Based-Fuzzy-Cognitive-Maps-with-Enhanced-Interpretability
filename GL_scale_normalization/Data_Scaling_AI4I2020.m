%% Data Scaling for Predictive Maintenance Dataset AI4I2020
clc
clear all;
warning("off")
%% Normalize each fold

%Number of groups for k-fold cross validation
numbfolds=10;

for k=1:numbfolds
disp(append(newline,num2str(k)," fold"));
%####################Read the datasets###########################    
%Paths for each training and test dataset
filepath_for_training = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(k),"\Training Dataset","\X_train_iter_",num2str(k),".csv");
filepath_for_test = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(k),"\Test Dataset","\X_test_iter_",num2str(k),".csv");

x_train = readtable(filepath_for_training);
x_train = table2array(x_train);

x_test = readtable(filepath_for_test);
x_test = table2array(x_test);

%################Scale Data by using GL Algorithm#######################
figure;
sgtitle(append(num2str(k)," fold"));
[scaled_x_train, setting] = glscale(x_train);
scaled_x_test = glscale(x_test,setting);


%##############Save the scaled datasets (for each fold)####################

%Paths for each training and test dataset
filepath_for_scaled_training = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(k),"\Training Dataset","\Scaled_X_train_iter_",num2str(k),".csv");
filepath_for_scaled_test = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\k-fold cross validation datasets\",num2str(k),"\Test Dataset","\Scaled_X_test_iter_",num2str(k),".csv");


writematrix(scaled_x_train,filepath_for_scaled_training,'Delimiter',',')
writematrix(scaled_x_test,filepath_for_scaled_test,'Delimiter',',')


end

%% Normalize the entire dataset

ai4i2020 = readtable("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\raw_data\ai4i2020_encoded_balanced.csv");
ai4i2020_dataset = table2array(ai4i2020);

X = ai4i2020_dataset(:,1:6);
y = ai4i2020_dataset(:,7:12);

figure;
sgtitle("Entire Dataset");
disp(newline)
[scaled_X, setting] = glscale(X);

scaled_dataset = [scaled_X,y];

filepath_for_scaled_dataset = append("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\raw_data\Scaled_ai4i2020_encoded_balanced.csv");
writematrix(scaled_dataset,filepath_for_scaled_dataset,'Delimiter',',')