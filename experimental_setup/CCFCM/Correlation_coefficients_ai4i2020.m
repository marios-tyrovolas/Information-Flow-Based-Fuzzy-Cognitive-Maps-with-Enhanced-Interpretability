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

%% Correlation coefficients

%R = corrcoef(A) returns the matrix of correlation coefficients for A, where the columns of A represent random variables and the rows represent observations
[R,P] = corrcoef(used_dataset);

indexes_for_p_values_smaller_than_0_05 = [];
deiktis=1;
for i=1:height(P)
    for j=1:width(P)
        if P(i,j)<0.05
            indexes_for_p_values_smaller_than_0_05(deiktis,1)=i;
            indexes_for_p_values_smaller_than_0_05(deiktis,2)=j;
            deiktis = deiktis +1;
        end
    end
end

W=[];
deiktis_new=1;
for i=1:height(R)
    for j=1:width(R)
        if (i==indexes_for_p_values_smaller_than_0_05(deiktis_new,1)) && (j==indexes_for_p_values_smaller_than_0_05(deiktis_new,2))
            W(i,j)=R(i,j);
            if deiktis_new < height(indexes_for_p_values_smaller_than_0_05)
                deiktis_new = deiktis_new +1;
            end
        end
    end
end

%% Save the matrix of normalized Ts and the corresponding confidence intervals

filepath_for_cor_coeff_weight_matrix = append("..\experimental_setup\CCFCM\weight_matrix_cor_coeff.csv");
writematrix(W,filepath_for_cor_coeff_weight_matrix,'Delimiter',',')