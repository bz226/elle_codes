clear;
close all;
clc;

%% DESCRIPTION:
%
% Get a random distribution of N phases over a number of points (e.g.
% unodes) with a certain percentage of points belonging to each phase and
% store them in a textfile
%
%% INPUT
%
resolution = 4; % number of points (e.g. unodes) will be resolution^2
number_of_phases = 2;
percentage_phases = [.8 .2]; % sum has to be 1
%
%% Start Calculating
%
%
% Set phases according to grain percentages, phases range from 0 to
% "number_of_phases":
%
GRID_PHASE = zeros(resolution^2,1);
%
% Store the previous position in GRID_PHASE during phase allocation:
%
storebegin = 1; 
storeend   = 0;
%
for i=1:number_of_phases
    storeend = storeend + round(percentage_phases(i)*(resolution^2));
    if storeend > resolution^2
        storeend = resolution^2;
    end
    GRID_PHASE( storebegin :storeend) = i-1;
    storebegin = storeend+1;
end
%
% Randomize:
%
GRID_PHASE = GRID_PHASE(randperm(resolution^2)');
%
%% Write to textfile
%
% Adapt it to Elle (phases counterd from 1 on, here for 2 phases only in
% output, the other one will be default)
%
ids = (0:1:(resolution^2)-1)';
GRID_PHASE = GRID_PHASE +1;
%
IDS = ids(GRID_PHASE==2);
PHASES = GRID_PHASE(GRID_PHASE==2);
%
% Use my function to write data:
%
write2file(['phase_distribution' sprintf('%03d',resolution) '.txt'],[IDS,PHASES]);
%
%% Plotting Possibility
%
break;
%
% Randomly distribute phases in "GRID_PHASE" and reshape it to square grid:
%
GRID_PHASE = GRID_PHASE(randperm(resolution^2)');
GRID_PHASE = reshape(GRID_PHASE,resolution,resolution);
%
% Generate a grid with IDs for each point (initially each point is a grain)
%
GRID_GRAINS = flip(reshape(...
    linspace(1,resolution^2,resolution^2),resolution,resolution...
                            )',1);
%
PlotGrids(GRID_PHASE,'phase');
%
