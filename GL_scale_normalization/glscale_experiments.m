%% Scripts for experiments on datasets with number of features less than 50
% In the experiments, each dataset will be will be scaled by using Min-max,
% Z-score, and GL algorithms; and then the scaled data will be used to
% train a logistic regression and a support vector machine classifier. The
% performances are evaluated by using AUC
% 
% Author: Xi Hang Cao
% Last updated: 3-24-2016
%%
clear
close all

addpath('utilities')

% A list of datasets will be used in this experiment
% NOTE: for illustration purposes, only a subset of all datasets is used.
% Users can add or remove datasets in the 'data' folder. 
datasetName = ...
    {...
    'BreastTissue',...
    'Diabetes',...
    'Parkinsons'...
    };

% Number of datasets
nFile = length(datasetName);

% Classifiers will be used in this experiment
classifierC = {'LR','SVM'};
nClassifier = length(classifierC);

% Scaling Methods will be used in this experiment
methodC = {'None','Minmax','Zscore','GL'};
nMethod = length(methodC);

% Place holder for the resutls
aucsC = cell(nFile,1);
auc_mean = nan(nClassifier,nMethod,nFile);


%%
   
for m = 1:nFile

    disp(['***************** ' datasetName{m} ' *****************'])
    load(['data/' datasetName{m}]);
    nS = size(data,1);
    
    pScore = zeros(nS,nMethod,nClassifier);
    label = data(:,1);

    origL = data(:,1);
    origF = data(:,2:end);

    % Generate CV indices 
    if nS <= 30
        nCV = nS;
        nRepeat = 1;
    else
        nCV = 5;
        nRepeat = 2; % number of repeats of the 5-fold cross-validation
    end
    
    aucs = nan(nClassifier,nMethod,nRepeat);
    
    for iRepeat = 1:nRepeat

        if nRepeat == 1
            ind = 1:nS;
        else
            ind = gencvind(origL, nCV);
        end
        
        trainIndC = cell(nCV,1);
        testIndC = cell(nCV,1);
        scoreC = cell(nCV,1);

        for cvInd = 1:nCV
            trainIndC{cvInd} = ind~=cvInd;
            testIndC{cvInd} = ind==cvInd;
        end
        
        for cvInd = 1:nCV
            
            for iMethod = 1:nMethod
                
                method = methodC{iMethod};
                trainInd = trainIndC{cvInd};
                testInd = testIndC{cvInd};

                F = origF(trainInd,:);
                labels = origL(trainInd,:);
                testF = origF(testInd,:);
                testLabels = origL(testInd,:);

                switch method
                    case 'None'

                    case 'Zscore'
                        % Global Z-score 
                        [F, nSetting] = normalize(F);
                        testF = zscore(testF,nSetting);
                    case 'Minmax'
                        % mapminmax
                        [tempF, PS] = mapminmax(F');
                        F = tempF';
                        tempF = mapminmax('apply',testF',PS);
                        testF = tempF';
                    case 'GL'
                        if (size(F,2)>50)
                            % if the number of features is greater than 50,
                            % we use the parallel version of the function.
                            % Note: parallel computing toolbox is required
                            [F,setting] = glscale_parallel(F);
                        else
                            [F,setting] = glscale(F);
                        end
                        testF = real(glscale(testF,setting));
                end
                tempy = labels;
                tempx = F;
                tempy = (tempy + 3)/2; % map (-1,1) to (1,2)
                
                % make the possitive class at the bottum
                [tempy,I] = sort(tempy);
                tempx = tempx(I,:);
                
                for iClassifier = 1:nClassifier
                    tStart = tic;
                    classifier = classifierC{iClassifier};
                    switch classifier
                        case 'LR'
                            lrMdl = MTlogiRegr(tempx,tempy*2-3);
                            pScore(testIndC{cvInd},iMethod,iClassifier) = logiEval(lrMdl,testF);
                        case 'SVM'
                            svmMdl = fitcsvm(tempx,tempy,'KernelFunction','linear');
                            [~,temp] = predict(svmMdl,testF);
                            pScore(testIndC{cvInd},iMethod,iClassifier) = temp(:,2);
                    end
                    tEnd = toc(tStart);
                    disp(['n = ' method ', c = ' classifier ', cv = ' num2str(cvInd) ', time = ' num2str(tEnd)])
                end
            end
        end
         
        pScore(isnan(pScore)) = 0;
        ol = origL;
        ol(ol==-1) = 0;
        
        for iMethod = 1:nMethod
            for iClassifier = 1:nClassifier
                tempScore = squeeze(pScore(:,iMethod,iClassifier));
                perf = binary_classification_performance_evaluation(ol,tempScore);
                aucs(iClassifier,iMethod,iRepeat) = perf.auROC;
            end
        end
    end
    aucsC{m} = aucs;
    auc_mean(:,:,m) = mean(aucs,3);
end










