clear;
close all;
clc;
addpath(genpath('functions/'));

%% Get eigenvalues using MTEX

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

%% END OF INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% Load data in ebsd object
%
ebsd = input_ice_data([pname fname]);
%
%% Extract individual orientations from ebsd object
%
% orients = get(ebsd('ice'),'orientations');
%
%% Vectors of c-axis in specimen coordinates and extract xyz
%
v=ebsd('ice').orientations*Miller(0,0,0,1,ebsd('ice').CS,'hkl');
[x,y,z]=double(v);

OT = 1./numel(x)*[x,y,z]'*[x,y,z];
[Vec,Diagonal]=eig(OT);
values=diag(Diagonal);
[values,index]=sort(values,'descend');
vec1(1:3)=Vec(:,index(1));
vec2(1:3)=Vec(:,index(2));
vec3(1:3)=Vec(:,index(3));
%
sum_values=sum(values);
values(1) = values(1)/sum_values;
values(2) = values(2)/sum_values;
values(3) = values(3)/sum_values;
disp(['E1 = ' num2str(values(1))]);
disp(['E2 = ' num2str(values(2))]);
disp(['E3 = ' num2str(values(3))]);


