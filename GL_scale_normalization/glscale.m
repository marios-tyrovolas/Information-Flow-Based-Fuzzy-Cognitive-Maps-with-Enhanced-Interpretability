%% GL Algorithm for robust data scaling 

% *Inputs*
% x:        feature matrix, where each row is an sample
% setting:  the setting of the normalization if performing scaling; when
%           learning the parameters, this variable needs to be absent.
%
% *Outputs*
% newx:     normalized x
% setting:  setting for the normalization.
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exampe:
%
% x_train = rand(20,10);
% x_test = rand(5,10);
% [scaled_x_train, setting] = glscale(x_train);
% scaled_x_test = glscale(x_test,setting);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Author: Xi Hang Cao
% Last update: 3-24-2016

%%
function [newx, setting] = glscale(x,setting)
newx = zeros(size(x));
nFea = size(x,2);

if nargin < 2   % When learning parameters is needed
    setting = zeros(4,nFea);    % initialization
    for iFea = 1:nFea   % for each feature
        tStart = tic;   % start time recording
        [tempf,tempz] = ecdf(x(:,iFea));  % find the ecef
        [tempz,i,~] = unique(tempz);    % find unique horizontal axis locations
        tempf = tempf(i);   % find corresponding vertial axis locations
        z = linspace(tempz(1),tempz(end),100);  % generate horizontal locations for 100 sample points
        f = interp1(tempz,tempf,z); % generate corresponding vertical locations
        setting(:,iFea) = generalLogiFit(z,f);  % find best fit parameters
        newx(:,iFea) = logiFunc(setting(:,iFea),x(:,iFea)); % scale the values
        tEnd = toc(tStart); % stop the time recording

% %%%%%%%%%%% you can uncomment this part to visualize the fitting        
        subplot(2,ceil(width(x)/2),iFea);
        plot(z,f,'ro','MarkerSize',10);
        hold on;
        plot(z,logiFunc(setting(:,iFea),z),'LineWidth',5,'Color','b');
        % hold off
        title(strcat('Feature # ',num2str(iFea)));
        xlabel('Original Values')
        ylabel('Scaled Values');
        legend('ecdf','Approximation using GL function','Location','southeast');
        set(gca,'FontSize',12)
        %close all;
        disp(['Feature # ' num2str(iFea) ' used ' num2str(tEnd)]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    end
else
    for iFea = 1:nFea
        newx(:,iFea) = logiFunc(setting(:,iFea),x(:,iFea));
    end
end