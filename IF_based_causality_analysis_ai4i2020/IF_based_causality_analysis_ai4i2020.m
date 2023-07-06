%%
clc
clear all
warning('off');
%% Read AI4I2020 Dataset

%Original dataset
ai4i2020 = readtable("..\dataset\raw_data\raw_ai4i2020_encoded.csv");
ai4i2020 = ai4i2020(:,1:7);

%Normalized dataset
%ai4i2020 = readtable("G:\.shortcut-targets-by-id\1-wapAl6N5YrCs68c4NiFKyvybXTXmdgZ\Ph_D_Tyrovolas\Our Papers\3rd_Paper-Proposal\Testbed Codes\AI4I_Case_Study\raw_data\Scaled_ai4i2020_feature_extraction.csv");

used_dataset = table2array(ai4i2020);


%% Plot process variables
for variable=1:width(used_dataset)
    subplot(width(used_dataset),1,variable)
    plot([1:length(used_dataset(:,variable))],used_dataset(:,variable));
    xlabel('Operation Cycles')
    title(ai4i2020.Properties.VariableNames{variable})
end

%% Information Flow between industrial process variables (AI4I2020) ("multi_causality_est_all_new" script)
disp('##################################################')
disp('IFs using the "multi_causality_est_all_new" script')
disp('##################################################')


[T,err90,err95,err99] = multi_causality_est_all_new(used_dataset, 1,1)

%% "multi_tau_est" script

%Calculate the IF rates from Xj->X1, j=1,....M for each time series variable
Normalizers=[];
for variable=1:width(used_dataset)
    matrix_at_each_iteration = [];
    %In each iteration each examined variable is the first column of the matrix_at
    %each_iteration and the rest ones are the other columns
    index=1;
    matrix_at_each_iteration(:,index)=used_dataset(:,variable);
    index=index+1;
    for j=1:width(used_dataset)
        if isequal(used_dataset(:,j), used_dataset(:,variable)) == false
            matrix_at_each_iteration(:,index)=used_dataset(:,j);
            index=index+1;
        end
    end
    
    
    [IFs_to_every_variable(:,variable), taus_to_every_variable(:,variable), dH1_star_to_every_variable(:,variable), dH1_noise_to_every_variable(:,variable)] = multi_tau_est(matrix_at_each_iteration, 1);
end

final_verified_T =[];
final_taus =[];
for variable=1:width(IFs_to_every_variable)
    index=2;
    for i=1:height(IFs_to_every_variable)
        if i==variable
            final_verified_T(i,variable)= NaN;
            final_taus(i,variable)= NaN;
        else
            final_verified_T(i,variable) = IFs_to_every_variable(index,variable);
            final_taus(i,variable) = taus_to_every_variable(index,variable);
            index=index+1;
        end
    end
end

final_verified_T = transpose(final_verified_T);
final_taus = transpose(final_taus);

disp('####################################')
disp('IFs using the "multi_tau_est" script')
disp('####################################')


for row = 1 : height(final_verified_T)
  fprintf('%7.4f ', final_verified_T(row, :));
  fprintf('\n');
end
fprintf('\n');
fprintf('\n');

disp('###############################################')
disp('Normalized IFs using the "multi_tau_est" script')
disp('###############################################')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% For convenience, let taus1 stores the normalized IF rate from Xj --> X1
%     j = 1, ..., M, (where taus1(1)=T11/Z,  taus1(2) = T21/Z, ...
%


for row = 1 : height(final_taus)
  fprintf('%7.4f ', final_taus(row, :));
  fprintf('\n');
end
fprintf('\n');
fprintf('\n');

%% Signficance Test
significant_Ts = [];
significant_taus = [];
for i=1:height(T)
    %a=0.01 significance test
    for j=1:width(T)
        lower_limit = T(i,j)-err99(i,j);
        upper_limit = T(i,j)+err99(i,j);
        if i==j
            significant_Ts(i,j) = 0;
            significant_taus(i,j) = 0;
        else
            if T(i,j) > 0
                if lower_limit >0 %The confidence interval takes only positive values
                    significant_Ts(i,j) = T(i,j);
                    significant_taus(i,j) = final_taus(i,j);
                else
                    significant_Ts(i,j) = 0;
                    significant_taus(i,j) = 0;
                end
            end
        
            if T(i,j) < 0
                if upper_limit < 0 %The confidence interval takes only negative values
                    significant_Ts(i,j) = T(i,j);
                    significant_taus(i,j) = final_taus(i,j);
                else
                    significant_Ts(i,j) = 0;
                    significant_taus(i,j) = 0;
                end
            end
        end
    end
    
end

%Normalizers for each parameter
for i=1:height(final_verified_T)
    for j=1:width(final_verified_T)
        if j~=i
            Z(i,j)=final_verified_T(i,j)./final_taus(i,j);
        else
            Z(i,j)=0;
        end
    end
end

confidence_intervals_for_significant_taus = cell(height(significant_taus),width(significant_taus));

for i=1:height(final_verified_T)
    for j=1:width(final_verified_T)
        if significant_taus(i,j)~=0
            significant_tau_upper_limit = significant_taus(i,j) + (err99(i,j)/Z(i,j));
            significant_tau_lower_limit = significant_taus(i,j) - (err99(i,j)/Z(i,j));
            confidence_intervals_for_significant_taus{i,j} = [significant_tau_lower_limit, significant_tau_upper_limit];
        else
            confidence_intervals_for_significant_taus{i,j} = 0;
        end
    end
end

%% Save the matrix of normalized Ts and the corresponding confidence intervals

filepath_for_tau_matrix = append("..\experimental_setup\IFFCM\IF_FCM_Learning\taus.csv");
writematrix(significant_taus,filepath_for_tau_matrix,'Delimiter',',')

save '..\experimental_setup\IFFCM\IF_FCM_Learning\confidence_intervals_taus.mat' confidence_intervals_for_significant_taus
