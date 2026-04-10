clear;
close all;
clc;
addpath(genpath('functions/'));

%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path to mtex files, have to be created usinf FS_elle2mtex first:
pname = [pwd '/mtexfiles/']; % make sure there is an "/" at the end

% name of the file that you wish to import (in the abovementioned folder)
% with ".txt" at the end
fname='DemoModel_mtex.txt';

% Initialize mtex (if necessary):
% Indicate the path where you installed MTEX:
mtexpath = '/home/florian/programs/mtex-4.1.3/'; % "/" needs to be at the end
run ([mtexpath 'install_mtex.m']);

%% END OF INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%% Reset default colormap
%
setMTEXpref('defaultColorMap',WhiteJetColorMap);
%
%% Plot pole figures from specific elle file 
%
% Elle file needs to be transfered to mtex input file using e.g. 
% FS_elle2mtex. 

% run input_ice_data.m % creates a variable with the data called "ebsd"
ebsd1 = input_ice_data([pname fname]);

odf_ice1 = calcODF(ebsd1('ice').orientations,'halfwidth',5*degree);

mod1 = Miller(0,0,0,1,ebsd1('ice').CS,'hkl');

setMTEXpref('FontSize',16);

figure; 
plotPDF(odf_ice1,mod1,'lower',...
    'resolution',1*degree,'colorrange',[0 10]);

set(gca, 'CLim', [0, 10]);