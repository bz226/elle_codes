clear;
close all;
% hold on;
clc; 
addpath(genpath('functions/'));

%% Plot grain size data
% 
% Requirement:
% Output file have been created by FS_statistics -u 2, fileroot has to be 
% put as prefix in front of the outfile name
%
%% INPUT % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
root = 'DemoModel'; % The start of the filename stored in "data/", should end with "_grainsizes_ice.txt"
line_format_specifier = 'k-'; % colour of the plotted line
%
% if 1st line of data is from input file with 0 percent deformation type 1,
% otherwise type 0 (than it is assumed that the first line is already the 
% 1st model step
firstline_initial_size = 0; 
incr_strain = 0.005; % input the incremental strain (or shear strain) used in the simulation
simpleshear = 0; % type 1 if simulation was simple shear, 0 if it was pure shear
timestep_sec = 1e8; % timestep used for simulation (attention, this is = number of DRX steps*Elle timestep)
initial_mean_area = 0;% Enter area in mm², if you do not want to use the normalization to initial area type 0
plottime=0; % Set to 1 to plot time on x-axis and 0 to plot strain on x-axis
%
%% END OF INPUT

disp('INFO: When using this script make sure you use the correct timestep setting!');

filename = ['data/' root '_grainsizes_ice.txt'];
%
[number_of_grains,area_fraction_of_phase,mean_grain_area,...
    mean_diameter_circ_grains,mean_diameter_square_grains,...
    ratio_circBYsquare_grains,mean_peri_ratio] = import_grainsize_data(filename);
%
%% Determine the strain history from number of steps and incr. strain:
lengthtmp = 1;
strain = zeros(1,size(number_of_grains,1));

if firstline_initial_size==1
    % Calculate with 1st data point has strain = 0
    for i=2:size(number_of_grains,1)
        lengthtmp = lengthtmp-(incr_strain*lengthtmp);
        strain(i) = (1-lengthtmp)*100;
    end
    time_sec = 0:timestep_sec:(size(number_of_grains,1)-1)*timestep_sec;
    steps = 0:1:(size(number_of_grains,1)-1);
else
    % Calculate with 1st data point has strain = 1 time incr_strain
    for i=1:size(number_of_grains,1)
        lengthtmp = lengthtmp-(incr_strain*lengthtmp);
        strain(i) = (1-lengthtmp)*100;
    end
    time_sec = timestep_sec:timestep_sec:(size(number_of_grains,1))*timestep_sec;
    steps = 1:1:(size(number_of_grains,1));    
end

if (simpleshear==1)
	strain=incr_strain.*steps;
end

%% Plot stuff
time_yrs = time_sec/(60*60*24*365.25);

if initial_mean_area==0
    areas4plot=mean_grain_area./1e-6;
else
    areas4plot=(mean_grain_area./1e-6)./initial_mean_area;
end

if (plottime==1)
    plot(time_yrs,areas4plot,line_format_specifier);
    xlim([0 max(time_yrs)]);
else
    plot(strain,areas4plot,line_format_specifier);
    xlim([0 max(strain)]);
end
    
ylim([min(areas4plot) max(areas4plot)]);
box on;
if (plottime==1)
    xlabel('time (years)');
else
    if (simpleshear==1)
        xlabel('shear strain');        
    else
        xlabel('vertical shortening (%)');
    end
end
if initial_mean_area==0
    ylabel('mean grain area (mm²)');
else
    ylabel('mean grain area / initial area');
end