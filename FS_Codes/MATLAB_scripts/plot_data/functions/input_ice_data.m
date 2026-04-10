function ebsd= input_ice_data(fname)
%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = {... 
  'not indexed',...
  crystalSymmetry('6/mmm', [4.5181 4.5181 7.356], 'X||a*', 'Y||b', 'Z||c', 'mineral', 'ice', 'color', 'light blue'),...
  'notIndexed'};

% plotting convention
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','outOfPlane');

%% Specify File Names

% path to files
% pname = [pwd '/mtexfiles/']; % make sure there is an "/" at the end

% which files to be imported
% fname = [pname 'fraction00rate09_step075.txt'];

%% Import the Data

% create an EBSD variable containing the data
ebsd = loadEBSD(fname,CS,'interface','generic',...
  'ColumnNames', { 'x' 'y' 'phi1' 'Phi' 'phi2' 'Phase'}, 'Bunge');
%%
% plot(ebsd)

% run plot_polefigure.m
end