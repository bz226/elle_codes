clear;
close all;
clc;
addpath(genpath('functions/'));

%% Plot slip system activities from output files 
% 
% Output file have been created by FS_statistics -i file.elle -u 3 1 -n
%
%% INPUT % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
root = 'DemoModel'; % The start of the filename stored in "data/", should end with "_grainsizes_ice.txt"
incr_strain = 0.005; % input the incremental strain (or shear strain) used in the simulation
simpleshear = 0; % type 1 if simulation was simple shear, 0 if it was pure shear
%
%% END OF INPUT

[basal,prism,pyram,sum] = import_slipsystemacts(['data/' root '_MeanSlipSysAct.txt']);

%% Determine the strain history from number of steps and incr. strain:
timestep = size(basal,1);
steps = 1:1:size(basal,1);
lengthtmp = 1;
strain = zeros(1,timestep);

for i=1:timestep
    lengthtmp = lengthtmp-(incr_strain*lengthtmp);
    strain(i) = (1-lengthtmp)*100;
end

if (simpleshear==1)
    strain = steps.*incr_strain;
end

hold on;
    plot(strain,basal);
    plot(strain,prism);
    plot(strain,pyram);
hold off;
ylim([0 1]);
xlim([strain(1) strain(end)]);
box on;

legend('basal','prismatic','pyramidal');

if (simpleshear==1)
    xlabel('shear strain');
else
    xlabel('vertical shortning (%)');
end
ylabel('Slip system activity');
title(['Data for model "' root '"'],'Interpreter','none');