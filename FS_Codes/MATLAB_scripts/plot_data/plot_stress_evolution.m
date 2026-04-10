clear;
close all;
clc;
addpath(genpath('functions/'));

%% Plot stress evolution from AllOutData
% 
% Requirement:
% Output files have been created by FS_statistics -i dummy.elle -u 1 1 -n
%
%% INPUT % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
root = 'DemoModel'; % The start of the filename stored in "data/", should end with "_AllOutData.txt"
incr_strain = 0.005; % input the incremental strain (or shear strain) used in the simulation
simpleshear = 0; % type 1 if simulation was simple shear, 0 if it was pure shear
%
%% END OF INPUT % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
%% Load data
%

[SVM,DVM,diffstress,stressFieldErr,strainrateFieldErr,basalact,prismact,pyramact,...
    s11,s22,s33,s23,s13,s12,d11,d22,d33,d23,d13,d12] = ...
        import_alloutdata(['data/' root '_AllOutData.txt']);
    
%
%% Determine the strain history from number of steps and incr. strain:
if simpleshear==1    
    steps = (1:size(SVM,1))';
    strain=incr_strain.*steps;
else
    timestep = size(SVM,1);
    lengthtmp = 1;
    strain = zeros(1,timestep);

    for i=1:timestep
        lengthtmp = lengthtmp-(incr_strain*lengthtmp);
        strain(i) = (1-lengthtmp)*100;
    end
end

hold on;
plot(strain,s11);
plot(strain,s22);
plot(strain,s33);
plot(strain,s12);
plot(strain,s13);
plot(strain,s23);

hold off;
xlim([strain(1) strain(end)]);

ylim([min(min([s11,s22,s33,s12,s13,s23])),...
        max(max([s11,s22,s33,s12,s13,s23]))]);
box on;
legend('s11','s22','s33','s12','s13','s23','Location','EastOutside');

if simpleshear==1    
    xlabel('shear strain');
else
    xlabel('vertical shortning (%)');
end
ylabel('stress in units of basal plane CRSS');
title(['Data for model "' root '"'],'Interpreter','none');